import numpy as np
import scipy as sp
import scipy.optimize as opt
import simulation_tools as simt
import einops

class multiSFS:
    """
    A class for storing mutation rate stratified polymorphism data and 
    performing inference of the non-decreasing latent SFS.
    """
    def __init__(self, data, neutral_sfs):
        """
        Initialize the WinSFS object.
        
        Parameters
        ----------
        data : array_like
            The observed polymorphism counts, stratified by mutation rate.
            (M x K) array, where M is the number of mutation rate classes and
            K is the number of frequency bins.
        neutral_sfs : array_like
            The observed neutral SFS polymorphism proportions, stratified by mutation rate.
            (M x K) array.
        """
        # assert that data and neutral_sfs are numpy arrays
        assert isinstance(data, np.ndarray)
        assert isinstance(neutral_sfs, np.ndarray)
        # assert that data and neutral_sfs are 2D arrays
        assert data.ndim == 2
        assert neutral_sfs.ndim == 2
        # assert that data and neutral_sfs have the same number of rows
        assert data.shape[0] == neutral_sfs.shape[0]
        # assert that data and neutral_sfs have the correct number of columns
        assert data.shape[1] == neutral_sfs.shape[1]

        # mut_offset = simt.multinomial_trans(neutral_sfs)
        
        # rename to match notation in methods, and store
        self.YY = data
        self.neutral_sfs = neutral_sfs
        # self.beta_0 = mut_offset
        # save a second copy of beta_0 with a column of zeros appended at the front
        # self.beta_0_0 = np.hstack((np.zeros((self.beta_0.shape[0], 1)), self.beta_0))

        # store dimensions
        self.M = self.YY.shape[0]
        self.K = self.YY.shape[1]

        # compute the total number of polymorphisms in each mutation rate class
        self.nn = np.sum(self.YY, axis=1)

        # # initialize the empty latent SFS inferred from traditional optimization
        # self.alpha_optim = None
        # # initialize the fit probabilities inferred from traditional optimization
        # self.fit_probs_optim = None

    def binarized_2bin(self, reverse = False):
        """
        Return binarized array of 0 for monomorphic sites and 1 for polymorphic sites.
        If reverse, 1 for monomorphic sites and 0 for polymorphic sites.
        
        Parameters
        ----------
        reverse : bool
            Determine whether to compute in reverse
        
        Returns
        -------
        binarized_array: array-like
            (M x 2) array.
        """
        #binary sfs for gnocchi
        assert self.K == 2
        binarized_array = np.zeros((self.M, self.K))

        if reverse:
            binarized_array[:,0] = 1
        else:
            binarized_array[:,1] = 1
            
        return binarized_array

    def neutral_cdf(self, reverse = False, zero_bin = False):
        """
        Compute the cdf of the neutral SFS
        
        Parameters
        ----------
        reverse : bool
            Determine whether to compue cdf in reverse (starting from monomorphic vs starting from most common bin)
            - if reverse, 0 bin equals 1
            - if not reverse most common bin equals 1
        
        Returns
        -------
        cdf: array-like
            The cdf neutral SFS.
            (M x K) array.
        """
        neutral_sfs_copy = np.copy(self.neutral_sfs)

        if reverse:
            for i in range(1, self.K):
                neutral_sfs_copy[:,:i] += einops.repeat(neutral_sfs_copy[:,i], 'h -> h repeat', repeat= i)

            if zero_bin:
                neutral_sfs_copy[:, -1] = 0
        else:
            for i in reversed(range(self.K)):
                neutral_sfs_copy[:,i+1:] += einops.repeat(neutral_sfs_copy[:,i], 'h -> h repeat', repeat= self.K - i - 1)

            if zero_bin:
                neutral_sfs_copy[:, 0] = 0

        
        
        return neutral_sfs_copy

    def zscore(self, distribution, reverse):
        """
        Compute the z score of some arbitrary distribution from neutral SFS.
        
        Parameters
        ----------
        distribution: (M x K) array

        Returns
        -------
        zscore : float
            The z-score of some distribution per-site.
        """

        # calculate tail log liklihood for the observed polymorphism
        data = einops.einsum(self.YY, distribution, "i j, i j -> i j").sum()

        # calculate expected log liklihood
        neutral_expected = einops.einsum(self.neutral_sfs, distribution, "i j, i j -> i j")
        neutral_expected = einops.reduce(neutral_expected, 'h w -> h', 'sum')
        neutral_expected_sum = einops.einsum(neutral_expected, self.nn, "i , i  -> i ").sum()

        # calculate variance of log likelihood
        neutral_expected_sq = einops.einsum(self.neutral_sfs, np.square(distribution), "i j, i j -> i j")
        neutral_expected_sq = einops.reduce(neutral_expected_sq, 'h w -> h', 'sum')
        neutral_var = neutral_expected_sq - np.square(neutral_expected) # variance as E[X^2] - E^2
        neutral_var_sum = einops.einsum(neutral_var, self.nn, "i , i  -> i ").sum()

        if reverse:
            return (data - neutral_expected_sum)/np.sqrt(neutral_var_sum), neutral_expected_sum, neutral_var_sum
        else:
            return (neutral_expected_sum - data)/np.sqrt(neutral_var_sum), neutral_expected_sum, neutral_var_sum
    
    def gnocchi(self):
        """
        Compute the gnocchi estimate
        
        Parameters
        ----------
        
        Returns
        -------
        gnocchi : float
            The score of gnocchi.
        """
        #binary sfs for gnocchi
        assert self.neutral_sfs.shape[1] == 2

        distribution = self.binarized_2bins()
        
        # calculate tail log liklihood for the observed polymorphism
        data = einops.einsum(self.YY, distribution, "i j, i j -> i j").sum()

        # calculate expected log liklihood
        neutral_expected = einops.einsum(self.neutral_sfs, distribution, "i j, i j -> i j")
        neutral_expected = einops.reduce(neutral_expected, 'h w -> h', 'sum')
        neutral_expected_sum = einops.einsum(neutral_expected, self.nn, "i , i  -> i ").sum()

        #gnocchi assumes poisson, so variance equals expectation
        neutral_var_sum = neutral_expected_sum

        return (neutral_expected_sum - data)/np.sqrt(neutral_var_sum)

    def extrainsight(self):
        """
        Compute the ExtRaINSIGHT estimate from dukler et al
        
        Parameters
        ----------
        
        Returns
        -------
        zscore : float
            The z-score of \lambda_s.
        """
        #binary sfs for extrainsight
        assert self.neutral_sfs.shape[1] == 2

        distribution = self.binarized_2bins()
        
        # calculate observed sum
        data = einops.einsum(self.YY, distribution, "i j, i j -> i j").sum()

        # calculate expectation
        neutral_expected = einops.einsum(self.neutral_sfs, distribution, "i j, i j -> i j")
        neutral_expected = einops.reduce(neutral_expected, 'h w -> h', 'sum')
        neutral_expected_sum = einops.einsum(neutral_expected, self.nn, "i , i  -> i ").sum()

        # lambda_s = (neutral_expected_sum - data)/neutral_expected_sum

        #calculate variance
        neutral_expected_sq = einops.einsum(self.neutral_sfs, np.square(distribution), "i j, i j -> i j")
        neutral_expected_sq = einops.reduce(neutral_expected_sq, 'h w -> h', 'sum')
        neutral_expected_sq_sum = einops.einsum(neutral_expected_sq, self.nn, "i , i  -> i ").sum()
        
        neutral_var_sum = neutral_expected_sum
        neutral_var_sum -= (neutral_expected_sum**2)*(neutral_expected_sq_sum)/(neutral_expected_sum**2)
        
        return (neutral_expected_sum - data)/np.sqrt(neutral_var_sum)

    def zscore_cdf(self, transformation = "log", reverse=False, zero_bin = False):
        """
        Compute the z score of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        
        Returns
        -------
        zscore : float
            The z-score of cdf per-site.
        """

        neutral_dist = self.neutral_cdf(reverse, zero_bin)
        
        if transformation == "log":
            transformed_dist = np.log(neutral_dist)
        elif transformation == "none":
            transformed_dist = neutral_dist
        else:
            print("transformation is not included")

        return self.zscore(transformed_dist, reverse)

    def zscore_binarized(self, reverse=False):
        """
        Compute the z score of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        
        Returns
        -------
        zscore : float
            The z-score of cdf per-site.
        """

        transformed_dist = self.binarized_2bin(reverse)
        
        return self.zscore(transformed_dist, reverse)
        

class multinomial_SFS_p:
    """
    A class for creating mutation rate stratified "p" as a representation of some probability under neutrality.
    """
    def __init__(self, neutral_sfs, transformation = "log", reverse = True):
        """
        Initialize the WinSFS object.
        
        Parameters
        ----------
        data : array_like
            The observed polymorphism counts, stratified by mutation rate.
            (M x K) array, where M is the number of mutation rate classes and
            K is the number of frequency bins.
        neutral_sfs : array_like
            The observed neutral SFS polymorphism proportions, stratified by mutation rate.
            (M x K) array.
        """
        # assert that data and mut_offset are numpy arrays
        assert isinstance(neutral_sfs, np.ndarray)
        # assert that data and mut_offset are 2D arrays
        assert neutral_sfs.ndim == 2

        self.reverse = reverse
        self.transformation = transformation
        
        self.neutral_sfs = neutral_sfs
        
        # store dimensions
        self.M = self.neutral_sfs.shape[0]
        self.K = self.neutral_sfs.shape[1]

        neutral_dist = self.neutral_cdf()
        if transformation == "log":
            self.transformed_dist = np.log(neutral_dist)
        elif transformation == "none":
            self.transformed_dist = neutral_dist
        else:
            print("transformation is not in the included list of transformations")

        self.expectation = einops.einsum(self.neutral_sfs, self.transformed_dist, "i j, i j-> i")
        expectation_sq = einops.einsum(self.neutral_sfs, np.square(self.transformed_dist), "i j, i j-> i")

        self.variance = expectation_sq - np.square(self.expectation)

    def neutral_cdf(self):
        """
        Compute the cdf of the neutral SFS
        
        Parameters
        ----------
        reverse : bool
            Determine whether to compue cdf in reverse (starting from monomorphic vs starting from most common bin)
        
        Returns
        -------
        cdf: array-like
            The cdf neutral SFS.
            (M x K) array.
        """

        # if reverse, monomorphic bin equals 1
        # if not reverse most common bin equals 1
        
        neutral_sfs_copy = np.copy(self.neutral_sfs)
        bin_num = self.K

        if self.reverse:
            for i in range(1, bin_num):
                neutral_sfs_copy[:,:i] += einops.repeat(neutral_sfs_copy[:,i], 'h -> h repeat', repeat= i)
        else:
            for i in reversed(range(bin_num)):
                neutral_sfs_copy[:,i+1:] += einops.repeat(neutral_sfs_copy[:,i], 'h -> h repeat', repeat= bin_num - i - 1)
        return neutral_sfs_copy

    def get_p(self, mu_index, AF_bin):
        """
        Compute the p of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        mu_index: int
        AF_bin: int
        
        Returns
        -------
        p : float
            (or log(p))
        """
        if self.reverse:
            return self.transformed_dist[mu_index, AF_bin]
        else:
            return -1 * self.transformed_dist[mu_index, AF_bin]

    def get_expected_p(self, mu_index):
        """
        Compute the Exp(p) of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        mu_index: int
        
        Returns
        -------
        p : float
            (or log(p))
        """
        if self.reverse:
            return self.expectation[mu_index]
        else:
            return -1 * self.expectation[mu_index]

    def get_var_p(self, mu_index):
        """
        Compute the Exp(p) of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        mu_index: int
        
        Returns
        -------
        p : float
            (or log(p))
        """
        return self.variance[mu_index]