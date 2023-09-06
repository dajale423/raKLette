import math
import glob
import os
import torch
import fastDTWF
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
    if offset is not None:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None]) - offset
    else:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None])
    return betas

def read_sfs_data(sfs_loc, prefix):
    """
    Read in the SFS data from a given folder and return the sfs_set for simPile

    Parameters
    ----------
    sfs_loc : str
        The location of the SFS data.
    prefix : str
        The prefix of the SFS files.

    Returns
    -------
    sfs_set : array_like
        The SFS data, (n_m x n_s x n_k) array, where n_m is the number of mutation
        rates, n_s is the number of selection coefficients, and n_k is the number of
        allele count classes.
    mu_vals : array_like
        The mutation rates.
    s_vals : array_like
        The selection coefficients.

    Notes
    -----
    Assumes that the SFS files are of the form prefix_sfs_s_{shet}_mu_{mu}.npy.
    """
    # get the SFS files
    sfs_files = glob.glob(os.path.join(sfs_loc, prefix + '_s_*.npy'))

    sfs_names = [os.path.basename(f) for f in sfs_files]
    sfs_names = [os.path.splitext(f)[0] for f in sfs_names]
    sfs_names = [f.split('_') for f in sfs_names]
    s_vals = np.array([float(f[3]) for f in sfs_names])
    mu_vals = np.array([float(f[5]) for f in sfs_names])

    # sort the files by mutation rate and then selection coefficient
    ind = np.lexsort((s_vals, mu_vals))
    sfs_files = [sfs_files[i] for i in ind]
    s_unique = np.unique(s_vals)
    mu_unique = np.unique(mu_vals)
    n_m = len(mu_unique)
    n_s = len(s_unique)
    # read in the first file to get the SFS dimensions
    sfs = np.load(sfs_files[0])
    # check that there is only one dimension for the SFS
    if len(sfs.shape) > 1:
        raise ValueError('SFS should be one-dimensional.')
    # check that the SFS has an even number of entries
    # and remove the last entry (fixed class) if so, assumes diploidy
    if sfs.shape[0] % 2 == 1:
        sfs = sfs[:-1]
        fixed = True
    n_k = sfs.shape[0]

    sfs_set = np.zeros((n_m, n_s, n_k))
    for ii in range(len(sfs_files)):
        jj = np.where(mu_unique == mu_vals[ind[ii]])[0][0]
        kk = np.where(s_unique == s_vals[ind[ii]])[0][0]
        sfs_set[jj,kk,:] = np.load(sfs_files[ii]) if not fixed else np.load(sfs_files[ii])[:-1]

    return sfs_set, mu_unique, s_unique

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
        # (this assumes diploidy)
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

    def sample(self, mu_vals, s_vals, bin=False):
        """
        Generate a sample of allele frequencies using the probability distribution
        implied by the SFS simulations.

        Parameters
        ----------
        mu_vals : array_like
            The mutation rates for each sample.
            (n) array.
        s_vals : array_like
            The selection coefficients for each sample.
            (n) array.
        bin : bool, optional
            Whether to use the SFS for sampling, if available. Default is False.

        Returns
        -------
        sample : array_like
            The sampled allele frequencies.
            (n) array.
        """

        # check that the shapes of the input arrays are consistent
        assert len(mu_vals) == len(s_vals)

        # get the nearest mutation rate and selection coefficient in the grid
        mu_ind = np.argmin(np.abs(self.mu_set[:, np.newaxis] - mu_vals), axis=0)
        s_ind = np.argmin(np.abs(self.shet_set[:, np.newaxis] - s_vals), axis=0)

        # get the SFS for the sampled mutation rates and selection coefficients
        if bin:
            # check if the SFS has been binned
            if self.binned_sfs is None:
                raise ValueError("SFS has not been binned.")
            sfs_cumsum = np.cumsum(self.binned_sfs, axis=-1)[mu_ind, s_ind, :]
        else:
            sfs_cumsum = np.cumsum(self.sfs_set, axis=-1)[mu_ind, s_ind, :]

        nn = len(mu_vals)
        random_values = np.random.rand(nn, 1)
        sample = np.sum(sfs_cumsum < random_values, axis=1)

        return sample


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
            if self.binned_sfs is None:
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
            if self.binned_sfs is None:
                raise ValueError("SFS has not been binned.")
            # Always get neutral betas in case binning has changed
            neutral_betas = self.get_neutral_betas(bin=True)
            betas = multinomial_trans(self.binned_sfs, offset=neutral_betas[:,None,:])
            self.betas_binned = betas
        else:
            if self.neutral_betas is None:
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
            count_range = (1, self.n_k)
        else:
            count_range = (0, self.n_k)

        # bin the SFS
        for i in np.arange(self.n_m):
            for j in np.arange(self.n_s):
                binned_sfs[i,j] = np.histogram(np.arange(0, self.n_k), bins=bins, 
                                               weights=self.sfs_set[i,j], 
                                               range=count_range)[0]

        # normalize the binned SFS to be a probability distribution in case zero-frequency class is excluded
        if not incl_zero:
            binned_sfs /= np.sum(binned_sfs, axis=2)[:,:,None]
        
        self.binned_sfs = binned_sfs
        self.bins = bins
        self.bin_means = bin_means(bins)
        self.bin_sizes = bin_sizes(bins)

        return binned_sfs
    
class betaDFE:
    """
    A class for generating samples of selection coefficients from a beta distributed DFE.

    Maybe doesn't make sense to have this as a class as the moment, but might be useful
    if we want to add more functionality later.
    """
    def __init__(self, a, b, min_shet=1e-5, max_shet=1):
        """
        Initialize the betaDFE object.

        Parameters
        ----------
        a : float
            The first parameter of the beta distribution.
        b : float
            The second parameter of the beta distribution.
        """
        self.a = a
        self.b = b
        self.min_shet = min_shet
        self.max_shet = max_shet

    def sample(self, nn):
        """
        Generate a sample of selection coefficients where the beta distribution is
        on a log scale from min_shet to max_shet.

        Parameters
        ----------
        nn : int
            The number of samples.

        Returns
        -------
        log10 shet : array_like
            The sampled selection coefficients.
            (nn) array.
        """
        shet = np.random.beta(self.a, self.b, size=nn)*(np.log10(self.max_shet)-np.log10(self.min_shet)) + \
            np.log10(self.min_shet)
        return shet

    def shet_to_x(self, shet):
        """
        Convert a selection coefficient to the beta distribution parameter.

        Parameters
        ----------
        shet : array_like
            The selection coefficients.

        Returns
        -------
        x : array_like
            The beta distribution parameter.
        """
        x = (np.log10(shet) - np.log10(self.min_shet))/(np.log10(self.max_shet) - np.log10(self.min_shet))
        return x
    
    def plot(self):
        """
        Plot the pdf and cdf of the beta distribution

        Returns
        -------
        fig : matplotlib.pyplot.figure
            The figure object.
        axes : matplotlib.pyplot.axes
            The axes object.
        """
        # make the axis ticks and labels a reasonable size
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        # set the x and y axis label sizes
        plt.rcParams['axes.labelsize'] = 16

        # make the legend a reasonable size
        plt.rcParams['legend.fontsize'] = 14

        shet_tick_locations = [10**exponent for exponent in 
                               range(int(np.log10(self.min_shet)), int(np.log10(self.max_shet)) + 1)]
        shet_tick_locations = [self.min_shet] + shet_tick_locations \
            if not self.min_shet in shet_tick_locations else shet_tick_locations
        shet_tick_locations =  shet_tick_locations + [self.max_shet] \
            if not self.max_shet in shet_tick_locations else shet_tick_locations
        shet_tick_locations = np.array(shet_tick_locations)
        x_tick_locations = self.shet_to_x(shet_tick_locations)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.linspace(0, 1, 10000)
        f_x = stats.beta.pdf(x, self.a, self.b)
        F_x = stats.beta.cdf(x, self.a, self.b)

        axes[0].plot(x, f_x)
        axes[0].set_xlabel(r'$s_{het}$')
        axes[0].set_ylabel(r'$f(\log_{10} s_{het})$')
        axes[0].set_title('beta pdf')
        axes[0].set_xticks(x_tick_locations)
        axes[0].set_xticklabels(shet_tick_locations)
        axes[0].set_xlim(-0.05, 1.05)

        axes[1].plot(x, F_x)
        axes[1].set_xlabel(r'$s_{het}$')
        axes[1].set_ylabel(r'$F(\log_{10} s_{het})$')
        axes[1].set_title('beta cdf')
        axes[1].set_xticks(x_tick_locations)
        axes[1].set_xticklabels(shet_tick_locations)
        axes[1].set_xlim(-0.05, 1.05)

        # show the alpha and beta values on one of the plots:
        axes[0].text(0.85, 0.95, r'$\alpha = $' + str(self.a), transform=axes[0].transAxes)
        axes[0].text(0.85, 0.9, r'$\beta = $' + str(self.b), transform=axes[0].transAxes)

        return fig, axes

class selectionSim:
    """
    A class for simulating allele frequencies under selection.
    """
    def __init__(self, sfs_pile, mu_dist):
        """
        Initialize the selectionSim object.

        Parameters
        ----------
        sfs_pile : simPile
            The simPile object containing the SFS simulations.
        mu_dist : function
            The distribution of mutation rates.

        Notes
        -----
        The mutation rate distribution should be a function that takes a single
        argument, the number of mutations, and returns a sample of mutation rates.
        """
        self.sfs_pile = sfs_pile
        self.mu_dist = mu_dist

    def make_sample(self):
        raise NotImplementedError
    
class windowSim(selectionSim):
    """
    A class for simulating allele frequencies under selection in a genomic window.
    """
    def __init__(self, sfs_pile, mu_dist, window_size, dfe):
        super().__init__(sfs_pile, mu_dist)
        self.window_size = window_size
        self.dfe = dfe

    def make_sample(self, n_windows):

        pass