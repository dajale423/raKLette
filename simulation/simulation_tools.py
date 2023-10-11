import math
import glob
import os
import sys
import torch
import fastDTWF
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

current_dir = os.path.dirname(os.path.abspath(__file__))
script_folder_path = os.path.join(current_dir, '..', 'scripts')
sys.path.append(script_folder_path)

import ml_raklette as mlr

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
    # bins = np.concatenate((bins[:-1], [math.ceil(nn/2), nn]))
    # Try including 50% and up in the last bin
    bins[-1] = nn-1
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

def simple_mu_dist(nn):
    mu_vals = np.array([1.e-09, 2.e-09, 5.e-09, 
                        1.e-08, 2.e-08, 5.e-08, 
                        1.e-07, 2.e-07,5.e-07])
    sim_mu_dist = [1e5,1e6,2e6,
                   1e6,1e5,1e3,
                   2e3,5e4,2e3]
    sim_mu_dist = sim_mu_dist / np.sum(sim_mu_dist)
    return np.random.choice(mu_vals, size=nn, p=sim_mu_dist)

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

        # verify that mu_set and shet_set are sorted, non-negative and in increasing order
        assert np.all(np.diff(self.mu_set) > 0)
        assert np.all(np.diff(self.shet_set) > 0)

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

        # Calculate and store cumulative sum of SFS to make binning faster
        self.sfs_cumsum = np.cumsum(self.sfs_set, axis=-1)

        self.binned_sfs = None
        self.bins = None
        self.bin_means = None
        self.bin_sizes = None

        self.neutral_betas = None
        self.betas = None

        self.neutral_betas_binned = None
        self.betas_binned = None

    def downsample(self, sample_size=2*70000, tv_sd=0.05, row_eps=1e-8):
        """
        Use fastDTWF to apply hypergeometric sampling to the SFS simulations.
        Create and return a new simPile object with the downsampled SFS simulations.
        """
        # loop through mutation rates and selection coefficients
        # and downsample each SFS
        new_sfs_set = np.zeros((self.n_m, self.n_s, sample_size))
        for ii in range(self.n_m):
            for jj in range(self.n_s):
                new_sfs = fastDTWF.hypergeometric_sample(torch.tensor(self.sfs_set[ii,jj,:]),
                                                         sample_size, tv_sd, row_eps, 
                                                         sfs=False).numpy()
                new_sfs[0] += new_sfs[-1] # add the fixed class to the zero-frequency class
                new_sfs_set[ii,jj,:] = new_sfs[:-1]  # remove the fixed class

        # create a new simPile object with the downsampled SFS simulations
        new_sim_pile = simPile(new_sfs_set, self.mu_set, self.shet_set)
        return new_sim_pile

    def sample(self, mu_vals, s_vals, bin_sfs=False, return_grid=False):
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
        bin_sfs : bool, optional
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
        if bin_sfs:
            # check if the SFS has been binned
            if self.binned_sfs is None:
                raise ValueError("SFS has not been binned.")
            sfs_cumsum = self.binned_sfs_cumsum[mu_ind, s_ind, :]
        else:
            sfs_cumsum = self.sfs_cumsum[mu_ind, s_ind, :]

        nn = len(mu_vals)
        random_values = np.random.rand(nn, 1)
        sample = np.sum(sfs_cumsum < random_values, axis=1)
        if return_grid:
            return sample, self.mu_set[mu_ind], self.shet_set[s_ind], mu_ind, s_ind
        return sample


    def get_neutral_betas(self, bin_sfs=False):
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
        if bin_sfs:
            # check if the SFS has been binned
            if self.binned_sfs is None:
                raise ValueError("SFS has not been binned.")
            neutral_betas = multinomial_trans(self.binned_sfs[:,self.neutral_index,:], offset=None)
            self.neutral_betas_binned = neutral_betas
        else:
            neutral_betas = multinomial_trans(self.sfs_set[:,self.neutral_index,:], offset=None)
            self.neutral_betas = neutral_betas
        return neutral_betas

    def get_betas(self, bin_sfs=False):
        """
        Get the multinomial coefficients for each selection coefficient at each mutation rate.

        Returns
        -------
        betas : array_like
            The multinomial coefficients for each selection coefficient at each mutation rate.
            (n_m x n_s x n_k-1) array.
        """
        if bin_sfs:
            # check if the SFS has been binned
            if self.binned_sfs is None:
                raise ValueError("SFS has not been binned.")
            # Always get neutral betas in case binning has changed
            neutral_betas = self.get_neutral_betas(bin_sfs=True)
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
        # make the bins, this part is relatively fast
        bins = make_bins(jmin, bb, self.n_k, incl_zero)

        # initialize the binned SFS
        binned_sfs = np.zeros((self.n_m, self.n_s, len(bins)-1))

        if incl_zero:
            binned_sfs = self.sfs_set[...,bins[:-1]] + \
                np.diff(self.sfs_cumsum[..., bins], axis=-1) - \
                    self.sfs_set[...,bins[1:]]
        else:
            binned_sfs = self.sfs_set[...,bins[:-1]] + \
                np.diff(self.sfs_cumsum[..., bins], axis=-1) - \
                    self.sfs_set[...,bins[1:]]
            binned_sfs -= self.sfs_set[...,0] # subtract the zero-frequency class
        binned_sfs[...,-1] += self.sfs_set[...,bins[-1]]

        # normalize the binned SFS to be a probability distribution in case zero-frequency class is excluded
        if not incl_zero:
            binned_sfs /= np.sum(binned_sfs, axis=2)[:,:,None]
        
        self.binned_sfs = binned_sfs
        self.binned_sfs_cumsum = np.cumsum(binned_sfs, axis=-1)
        self.bins = bins
        self.bin_means = bin_means(bins)
        self.bin_sizes = bin_sizes(bins)

        return binned_sfs

class neutralDFE:
    """
    Class for generating samples of selection coefficients from a neutral DFE.
    
    Only really useful because some functions need a DFE object.
    """
    def __init__(self):
        """
        Initialize the neutralDFE object.
        """
        pass
    
    def grid_pmf(self, shet_grid):
        """
        Neutral pmf is a point mass at zero.
        """
        return np.where(shet_grid==0, 1, 0)
    
    def sample(self, nn):
        """
        Neutral DFE samples are all zero (log scale).
        """
        return -np.inf*np.ones(nn)

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

    def pdf(self, shet):
        """
        Compute the pdf of the beta distribution.

        Parameters
        ----------
        shet : array_like
            The selection coefficients.

        Returns
        -------
        pdf : array_like
            The pdf of the beta distribution.
        """
        x = self.shet_to_x(shet)
        pdf = stats.beta.pdf(x, self.a, self.b)
        return pdf
    
    def cdf(self, shet):
        """
        Compute the cdf of the beta distribution.

        Parameters
        ----------
        shet : array_like
            The selection coefficients.

        Returns
        -------
        cdf : array_like
            The cdf of the beta distribution.
        """
        x = self.shet_to_x(shet)
        cdf = stats.beta.cdf(x, self.a, self.b)
        return cdf
    
    def grid_pmf(self, shet_grid):
        """
        Compute the pmf of the beta distribution on a grid of selection coefficients.

        Parameters
        ----------
        shet_grid : array_like
            The selection coefficients grid.

        Returns
        -------
        pmf : array_like
            A pmf on the shet grid that approximates the beta distribution.
        """
        x_grid = self.shet_to_x(shet_grid)
        x_mids = (x_grid[1:] + x_grid[:-1])/2
        x_mids = np.concatenate([[0], x_mids, [1]])
        x_cdf = stats.beta.cdf(x_mids, self.a, self.b)
        pmf = np.diff(x_cdf)
        return pmf

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
        shet = np.random.beta(self.a, self.b, size=nn)*(np.log10(self.max_shet) -
                                                        np.log10(self.min_shet)) + \
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
        x = (np.log10(shet) - np.log10(self.min_shet))/(np.log10(self.max_shet) - 
                                                        np.log10(self.min_shet))
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
        plt.rcParams['axes.labelsize'] = 20

        # make the legend a reasonable size
        plt.rcParams['legend.fontsize'] = 14
        plt.tight_layout()

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
    def __init__(self, sfs_pile, mu_dist, window_size, dfe, 
                 neut_p=0, bin_sfs=True, reference_mu=1e-8):
        super().__init__(sfs_pile, mu_dist)
        self.window_size = window_size
        self.dfe = dfe
        self.neut_p = neut_p
        self.bin_sfs = bin_sfs
        self.reference_mu = reference_mu
        self.mu_ind = np.argmin(np.abs(self.sfs_pile.mu_set - self.reference_mu))

        # check that the sim_pile has a neutral simulation if neut_p > 0
        if self.neut_p > 0 and not self.sfs_pile.has_neutral:
            raise ValueError("No neutral simulation in the set.")
        
        self.dfe_sfs = self.dfe_sfs()
        if bin_sfs:
            self.neut_sfs = self.sfs_pile.binned_sfs[self.mu_ind, self.sfs_pile.neutral_index, :]
        else:
            self.neut_sfs = self.sfs_pile.sfs_set[self.mu_ind, self.sfs_pile.neutral_index, :]
        self.KL = self.calc_dfe_KL()

    def get_neutral_betas(self):
        return self.sfs_pile.neutral_betas_binned if self.bin_sfs else self.sfs_pile.neutral_betas

    def dfe_sfs(self):
        """
        Get the SFS for the provided dfe by integrating out selection coefficients

        Returns
        -------
        dfe_sfs : array_like
            The SFS for the provided dfe.
            (n_m, n_k) array, where n_m is the number of mutation rates and n_k is
            the number of allele count classes.
        """
        s_vals = self.sfs_pile.shet_set
        s_vals = s_vals[s_vals > 0]
        s_pmf = self.dfe.grid_pmf(s_vals)
        # if self.neut_p > 0:
        s_pmf = (1-self.neut_p)*s_pmf
        s_pmf = np.concatenate([[self.neut_p], s_pmf])
        if self.bin_sfs:
            dfe_sfs = np.sum(self.sfs_pile.binned_sfs*s_pmf[:,None], axis=1)
        else:
            dfe_sfs = np.sum(self.sfs_pile.sfs_set*s_pmf[:,None], axis=1)
        self.dfe_sfs = dfe_sfs
        return dfe_sfs

    def calc_dfe_KL(self):
        """
        Compute the KL divergence between the SFS under the given DFE and the
        SFS under neutrality, defined by the sfs_pile.
        """
        # Take into account where dfe_sfs is zero, by removing these entries
        # x log(x) -> 0 as x -> 0
        dfe_sfs = self.dfe_sfs[self.mu_ind]
        neut_sfs = self.neut_sfs[dfe_sfs > 0]
        dfe_sfs = dfe_sfs[dfe_sfs > 0]
        KL = np.sum(dfe_sfs*np.log(dfe_sfs/neut_sfs))
        return KL

    def make_sample(self, n_windows):
            """
            Generate a sample of simulated data for a given number of windows.

            Parameters:
            n_windows (int): The number of windows to generate data for.

            Returns:
            tuple: A tuple containing four arrays:
                - A 3D array of shape (n_windows, n_mu, n_shet), 
                  where n_mu and n_shet are the number of mutation rates and selection coefficients respectively. 
                  This array contains the simulated allele frequencies for each window.
                - A 1D array of length n_windows containing the KL divergence between the DFE and neutrality for each window.
                - A 2D array of shape (n_windows, window_size) containing the mutation rates for each window.
                - A 2D array of shape (n_windows, window_size) containing the selection coefficients for each window.
            """
            result = np.zeros((n_windows,) + self.dfe_sfs.shape, dtype=int)
            result_KL = np.zeros(n_windows)
            result_shet = np.zeros((n_windows, self.window_size))
            result_mu = np.zeros((n_windows, self.window_size))
            #TODO: somehow speed this up by removing the loop
            for ii in range(n_windows):
                # sample a set of mutation rates
                mu_vals = self.mu_dist(self.window_size)
                # sample a set of selection coefficients
                shet_vals = np.power(10, self.dfe.sample(self.window_size))
                # sample allele frequencies
                freqs, mu_grid, shet_grid, _, shet_ind = self.sfs_pile.sample(mu_vals, shet_vals, 
                                                                 bin_sfs=self.bin_sfs, return_grid=True)
                result[ii] = reshape_counts(freqs, mu_grid, self.sfs_pile.mu_set, 
                                            len(self.sfs_pile.bin_means))
                # compute the KL divergence between the DFE and neutrality
                # count the number of times each shet value is sampled
                shet_counts = np.bincount(shet_ind, minlength=len(self.sfs_pile.shet_set))
                sample_shet_dist = shet_counts/np.sum(shet_counts)
                # get the KL divergence between the sampled shet distribution and the DFE
                if self.bin_sfs:
                    dfe_sfs = np.sum(self.sfs_pile.binned_sfs[self.mu_ind]*
                                     sample_shet_dist[:,None], axis=0)
                else:
                    dfe_sfs = np.sum(self.sfs_pile.sfs_set[self.mu_ind]*
                                     sample_shet_dist[:,None], axis=0)
                neut_sfs = self.neut_sfs[dfe_sfs>0]
                dfe_sfs = dfe_sfs[dfe_sfs>0]
                result_KL[ii] = np.sum(dfe_sfs*np.log(dfe_sfs/neut_sfs))
                result_shet[ii] = shet_grid
                result_mu[ii] = mu_grid
            return result, result_KL, result_mu, result_shet
    
def reshape_counts(freq_bins, mu_grid, mu_vals, n_k):
    """
    Reshape the allele frequency counts into a 2D array, where the first dimension
    is the mutation rate and the second dimension is the allele frequency.

    Parameters
    ----------
    freq_bins : array_like
        The allele frequencies
        n_l array, where n_l is the number of simulated sites
    mu_grid : array_like
        The mutation rates
        n_l array, where n_l is the number of simulated sites
    max_bin : int
        The maximum allele frequency bin.

    Returns
    -------
    freq_counts : array_like
        The allele frequency counts.
        (n_m x n_k) array, where n_m is the number of mutation rates and n_k is
        the number of allele frequency bins.
    """
    # check that all mu_grid are in mu_vals
    assert np.all(np.isin(mu_grid, mu_vals))

    # result = np.zeros((len(mu_vals), n_k), dtype=int)
    # for ii in range(len(mu_vals)):
    #     result[ii] = np.bincount(freq_bins[mu_grid==mu_vals[ii]], minlength=n_k)
    # return result

    mask = mu_grid[:, np.newaxis] == mu_vals

    # Use broadcasting to count occurrences for each mu_val
    result = np.sum(mask[..., np.newaxis] * (freq_bins[:, np.newaxis, np.newaxis] == np.arange(n_k)), 
                    axis=0)

    # Convert the result to integer type if needed
    result = result.astype(int)
    return result

#TODO: what is "counts" supposed to be doing here?
def calc_comparison(window_sim, win_sfs, counts):
    """
    Calculate comparison metrics between the window simulation and the fitted
    maximum likelihood non-decreasing SFS

    Parameters
    ----------
    window_sim : windowSim
        The window simulation object.
    win_sfs : ml_raklette.WinSFS
        The fitted maximum likelihood non-decreasing SFS.
    counts : array_like
        The allele frequency counts.
        (n_m x n_k) array, where n_m is the number of mutation rates and n_k is
        the number of allele frequency bins.

    Returns
    -------
    result : dict
        A dictionary containing the comparison metrics.
    """
    # verify that the ml SFS was estimated
    assert win_sfs.fit_probs_optim is not None
    # verify that the window simulation and the WinSFS have the same number of mutation rates
    assert window_sim.dfe_sfs.shape[0] == win_sfs.fit_probs_optim.shape[0]

    # get the KL divergence between the fitted SFS and neutrality
    # using the reference mutation rate given for the window simulation
    KL_fit = win_sfs.KL(window_sim.mu_ind)
    KL_dfe = window_sim.KL

    result = {"KL_fit": KL_fit, 
              "KL_dfe": KL_dfe}
    return result

def plot_KL(KL_set, KL_set_neut, KL_sim, ax):
    """
    Plots KL divergence values for a given set of data.

    Parameters:
    KL_set (numpy.ndarray): Array of KL divergence values.
    KL_set_neut (numpy.ndarray): Array of neutral KL divergence values.
    KL_sim (numpy.ndarray): Array of true simulated KL divergence values.
    ax (matplotlib.axes.Axes): Axes object to plot on.

    Returns:
    None
    """
    
    bin_sizes_plot = ["max"] + bin_sizes[1:]
    ax.violinplot(KL_set_neut.T, positions=np.arange(1, len(bin_sizes_plot)+1),
            widths=0.6, showmedians=True, showextrema=False);
    # ax.boxplot(KL_set_neut.T, positions=np.arange(1, len(bin_sizes_plot)+1),
        #    widths=0.6, patch_artist=True, boxprops=dict(facecolor="red", alpha=0.5),
        #    medianprops=dict(color="k"));
    #ax.boxplot(KL_set.T);
    ax.violinplot(KL_set.T, positions=np.arange(1, len(bin_sizes_plot)+1),
            widths=0.6, showmedians=True, showextrema=False);
    
    ax.plot(np.arange(1, len(bin_sizes_plot)+1), np.mean(KL_sim,-1));
    # label the x-axis ticks
    ax.set_xticks(range(1, len(bin_sizes_plot)+1), bin_sizes_plot, rotation=90); 
    # label the y-axis
    ax.set_ylabel("KL divergence");
    # label the x-axis
    ax.set_xlabel("bin size");
    ax.set_yscale("symlog", linthresh=2e-3)
    # increase the size of the y-axis ticks
    ax.tick_params(axis='y', which='major', labelsize=16)
    # increase the size of the x-axis ticks
    ax.tick_params(axis='x', which='major', labelsize=16)
    # add more ticks to the y-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

def run_test(sim_pile_test, sample_size, dfe, n_sites=100000, bin_sizes=bin_sizes, neut_p=0):
    """
    Runs a simulation test using the provided parameters and returns the KL divergence values.

    Args:
    - sim_pile_test: a SimPile object representing the simulated data
    - sample_size: an integer representing the number of samples to take
    - dfe: a DFE object representing the distribution of fitness effects
    - n_sites: an integer representing the number of sites to simulate
    - bin_sizes: a list of integers representing the bin sizes to use
    - neut_p: a float representing the proportion of neutral sites

    Returns:
    - KL_set: a numpy array of shape (len(bin_sizes), sample_size) representing the KL divergence values for each sample and bin size
    - KL_sim: a numpy array of shape (len(bin_sizes), sample_size) representing the KL divergence values for each sample and bin size using the simulated data
    """
    
    KL_set = np.zeros((len(bin_sizes), sample_size))
    KL_sim = np.zeros((len(bin_sizes), sample_size))
    for ii in range(len(bin_sizes)):
        bin_test = sim_pile_test.bin_sfs(1, bin_sizes[ii])
        bin_betas_test = sim_pile_test.get_betas(bin_sfs=True)

        window_test = windowSim(sim_pile_test, simple_mu_dist, n_sites, 
                                dfe, neut_p=neut_p, bin_sfs=True)
        counts, KL, _, _ = window_test.make_sample(sample_size)
        for jj in range(sample_size):
            win_sfs = mlr.WinSFS(counts[jj], window_test.get_neutral_betas())
            win_sfs.ml_optim(jac=True, beta_max=100, verbose=False)
            KL_compare = calc_comparison(window_test, win_sfs, counts[jj])
            KL_set[ii, jj] = KL_compare["KL_fit"]
        KL_sim[ii, :] = KL
    return KL_set, KL_sim