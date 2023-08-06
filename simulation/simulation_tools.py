import math
import torch
import fastDTWF
import numpy as np

# convert list of value to a list of indices where the value changes 
# and what those values are
def get_switch_points(x):
    switch_points = []
    switch_values = []
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            switch_points.append(i)
            switch_values.append(x[i])
    switch_points = np.array(switch_points, dtype=int)
    switch_points = -switch_points + switch_points[-1] + 1
    switch_points = np.concatenate([switch_points, [0]])
    switch_values = np.array(switch_values, dtype=int)
    switch_values = np.concatenate([[x[0]], switch_values])
    return switch_points, switch_values

# Tennessen et al. 2012 model of European population size
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3312577/
def tennessen_model(kk=1):
    N0 = math.ceil(7310*kk)
    N_old_growth = math.ceil(14474*kk)
    N_ooa_bn = math.ceil(1861*kk)
    N_ooa_bn_2 = math.ceil(1032*kk)
    N_growth_1 = math.ceil(9300*kk)
    N_growth_2 = math.ceil(512000*kk)

    t_old_growth = math.ceil(3880*kk)
    t_ooa_bn = math.ceil(1120*kk)
    t_growth_1 = math.ceil(715*kk)
    t_growth_2 = math.ceil(205*kk)

    r_growth_1 = (N_growth_1/N_ooa_bn_2)**(1/(t_growth_1-1))
    r_growth_2 = (N_growth_2/N_growth_1)**(1/(t_growth_2))

    N_set = np.array([N0] + [N_old_growth]*t_old_growth)
    N_set = np.append(N_set, [N_ooa_bn]*t_ooa_bn)
    N_set = np.append(N_set, N_ooa_bn_2*r_growth_1**np.arange(t_growth_1))
    N_set = np.append(N_set, N_growth_1*r_growth_2**np.arange(1, t_growth_2+1))
    return N_set.astype(np.int)

def tenn_sfs(shet, mu):
    """
    Generate the SFS for the Tennessen et al. 2012 model of European population size

    Parameters
    ----------
    shet : float
        The selection coefficient
    mu   : float
        The mutation rate
    
    Returns
    -------
    tenn_sfs : array_like
        The SFS for the Tennessen et al. 2012 model of European population size
    """
    tenn_N = tennessen_model()
    switch_points_tenn, pop_size_list_tenn = get_switch_points(tenn_N)
    shet = torch.tensor(shet, dtype=torch.float64)
    mu = torch.tensor(mu, dtype=torch.float64)
    tenn_sfs = fastDTWF.get_likelihood(
        pop_size_list=pop_size_list_tenn*2,    # population size at present
        switch_points=switch_points_tenn,      # sample individuals at present 
        sample_size=pop_size_list_tenn[-1]*2,  # sample the whole population
        s_het=shet,                            # selection coefficient
        mu_0_to_1=mu,                          # rate of mutating from "0" to "1"
        mu_1_to_0=mu,                          # rate of mutating from "1" to "0"
        dtwf_tv_sd=0.1,                        # controls accuracy of approximation
        dtwf_row_eps=1e-8,                     # controls accuracy of approximation
        sampling_tv_sd=0.05,                   # controls accuracy of approximation
        sampling_row_eps=1e-8,                 # controls accuracy of approximation
        no_fix=True,                           # condition on non-fixation
        sfs=False,
        injection_rate=0.0,
        use_condensed=True
    )
    return tenn_sfs

def make_bins(jmin, bb, nn, incl_zero=False):
    """
    Make a set of bins for the SFS.

    Parameters
    ----------
    jmin : int
        The minimum bin size.
    bb : float
        The bin size multiplier.
    nn : int
        The sample size.
    incl_zero : bool
        Whether to include a bin for the zero-frequency class.

    Returns
    -------
    bins : array_like
        The set of bins.
    """
    bins = np.arange(1, jmin + 1, dtype=int) # starting bins
    bb_next = bb
    while bins[-1] > bb_next: # catch up past starting bins
        bb_next *= bb
    bb_next = math.ceil(bb_next) if math.floor(bb_next)<=bins[-1] else math.floor(bb_next)
    bins = np.append(bins, bb_next)
    while bins[-1] < nn/2:
        bb_next *= bb
        bb_next = math.ceil(bb_next) if math.floor(bb_next)==bins[-1] else math.floor(bb_next)
        bins = np.append(bins, bb_next)
    bins = np.concatenate((bins[:-1], [math.ceil(nn/2), nn]))
    if incl_zero:
        bins = np.concatenate(([0], bins))
    return bins

def bin_means(bins):
    """
    Compute bin means as the mean allele count within each bin.

    Parameters
    ----------
    bins : array_like
        The set of bins.

    Returns
    -------
    result : array_like
        The bin means.
    """
    result = np.zeros(len(bins)-1)
    for ii in range(1, len(bins)):
        result[ii-1] = np.mean(np.arange(bins[ii-1], bins[ii]))
    return result

def bin_sizes(bins):
    """
    Compute bin sizes as the number of allele counts within each bin.
    
    Parameters
    ----------
    bins : array_like
        The set of bins.

    Returns
    -------
    result : array_like
        The bin sizes.
    """
    result = np.zeros(len(bins)-1)
    for ii in range(1, len(bins)):
        result[ii-1] = bins[ii] - bins[ii-1]
    return result

def multinomial_trans(sfs_probs, offset=None):
    """
    Convert a set of SFS probabilities to multinomial coefficients.

    Parameters
    ----------
    sfs_probs : array_like
        The SFS probabilities. Last dimension should be a probability distribution
        over the allele count classes.
    offset : array_like
        The neutral offset to subtract from each SFS probability. If None, no offset
        is subtracted.

    Returns
    -------
    betas : array_like
        The multinomial coefficients.
    """
    sfs_probs = np.array(sfs_probs)
    P_0 = sfs_probs[...,0]
    if offset:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None]) - offset
    else:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None])
    return betas

class simPile:
    """
    A class for storing and using SFS simulations over a grid of selection
    coefficients and mutation rates.
    """
    def __init__(self, sfs_set, mu_set, shet_set):
        """
        Initialize the simPile object.

        Parameters
        ----------
        sfs_set : array_like
            The SFS for each selection coefficient and mutation rate in the grid.
            (n_m x n_s x n_k) array. This is written with the output of fastDTWF
            in mind, so n_k should be 2*N, where N is the sample or population
            size and the first entry is the zero-frequency class. We do not include 
            the fixed class in the SFS. We check this by veryfying that n_k is even.
            We will normalize in case the provided SFS is not a probability distribution.
        mu_set : array_like
            The mutation rates for each simulation in the grid.
            (n_m) array.
        shet_set : array_like
            The selection coefficients for each simulation in the grid.
            (n_s) array.
        """

        self.sfs_set = sfs_set
        self.shet_set = shet_set
        self.mu_set = mu_set
        self.n_m, self.n_s, self.n_k = self.sfs_set.shape
        self.ac_set = np.arange(0, self.n_k) # allele count set

        # verify that the shapes of the input arrays are consistent
        assert self.n_m == len(self.mu_set)
        assert self.n_s == len(self.shet_set)

        # verify that the SFS has an even number of bins
        # meaning it is zero-inclusive and does not include the fixed class
        assert self.n_k % 2 == 0

        # normalize the SFS
        self.sfs_set /= np.sum(self.sfs_set, axis=2)[:,:,None]

        # check if there is a neutral simulation
        self.has_neutral = np.sum(self.shet_set==0) > 0
        # get the index of the neutral simulation
        self.neutral_index = np.where(self.shet_set==0)[0][0] if self.has_neutral else None

        self.binned_sfs = None
        self.bins = None
        self.bin_means = None
        self.bin_sizes = None

        self.neutral_betas = None
        self.betas = None

        self.neutral_betas_binned = None
        self.betas_binned = None

        def get_neutral_betas(self, bin=False):
            """
            Get the multinomial coefficients for neutral simulations at each mutation rate.

            Returns
            -------
            neutral_betas : array_like
                The multinomial coefficients for neutral simulations at each mutation rate.
                (n_m x n_k-1) array.
            """
            if not self.has_neutral:
                raise ValueError("No neutral simulation in the set.")
            if bin:
                # check if the SFS has been binned
                if not self.binned_sfs:
                    raise ValueError("SFS has not been binned.")
                neutral_betas = multinomial_trans(self.binned_sfs[:,self.neutral_index,:], offset=None)
                self.neutral_betas_binned = neutral_betas
            else:
                neutral_betas = multinomial_trans(self.sfs_set[:,self.neutral_index,:], offset=None)
                self.neutral_betas = neutral_betas
            return neutral_betas

        def get_betas(self, bin=False):
            """
            Get the multinomial coefficients for each selection coefficient at each mutation rate.

            Returns
            -------
            betas : array_like
                The multinomial coefficients for each selection coefficient at each mutation rate.
                (n_m x n_s x n_k-1) array.
            """
            if bin:
                # check if the SFS has been binned
                if not self.binned_sfs:
                    raise ValueError("SFS has not been binned.")
                if not self.neutral_betas_binned:
                    neutral_betas = self.get_neutral_betas(bin=True)
                else:
                    neutral_betas = self.neutral_betas_binned
                betas = multinomial_trans(self.binned_sfs, offset=neutral_betas[:,None,:])
                self.betas_binned = betas
            else:
                if not self.neutral_betas:
                    neutral_betas = self.get_neutral_betas()
                else:
                    neutral_betas = self.neutral_betas
                betas = multinomial_trans(self.sfs_set, offset=neutral_betas[:,None,:])
                self.betas = betas
            return betas            

        def bin_sfs(self, jmin, bb, incl_zero=True):
            """
            Bin the SFS.

            Parameters
            ----------
            jmin : int
                The minimum bin size.
            bb : float
                The bin size multiplier.
            incl_zero : bool
                Whether to include a bin for the zero-frequency class.

            Returns
            -------
            binned_sfs : array_like
                The binned SFS as a probability distribution.
            """
            # make the bins
            bins = make_bins(jmin, bb, self.n_k, incl_zero)

            # initialize the binned SFS
            binned_sfs = np.zeros((self.n_m, self.n_s, len(bins)-1))

            if not incl_zero:
                range = (1, self.n_k)
            else:
                range = (0, self.n_k)

            # bin the SFS
            for i in range(self.n_m):
                for j in range(self.n_s):
                    binned_sfs[i,j] = np.histogram(np.arange(0, self.n_k), bins=bins, 
                                                   weights=self.sfs_set[i,j], range=range)[0]

            # normalize the binned SFS to be a probability distribution in case zero-frequency class is excluded
            if not incl_zero:
                binned_sfs /= np.sum(binned_sfs, axis=2)[:,:,None]
            
            self.binned_sfs = binned_sfs
            self.bins = bins
            self.bin_means = bin_means(bins)
            self.bin_sizes = bin_sizes(bins)

            return binned_sfs
        