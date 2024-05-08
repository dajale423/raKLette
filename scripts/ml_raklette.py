import numpy as np
import scipy as sp
import scipy.optimize as opt
import simulation_tools as simt
import einops

class WinSFS:
    """
    A class for storing mutation rate stratified polymorphism data and 
    performing maximum likelihood inference of the non-decreasing latent SFS.
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
        # assert that data and mut_offset are numpy arrays
        assert isinstance(data, np.ndarray)
        assert isinstance(neutral_sfs, np.ndarray)
        # assert that data and mut_offset are 2D arrays
        assert data.ndim == 2
        assert neutral_sfs.ndim == 2
        # assert that data and mut_offset have the same number of rows
        assert data.shape[0] == neutral_sfs.shape[0]
        # assert that data and mut_offset have the correct number of columns
        assert data.shape[1] == neutral_sfs.shape[1]

        mut_offset = simt.multinomial_trans(neutral_sfs)
        
        # rename to match notation in methods, and store
        self.YY = data
        self.neutral_sfs = neutral_sfs
        self.beta_0 = mut_offset
        # save a second copy of beta_0 with a column of zeros appended at the front
        self.beta_0_0 = np.hstack((np.zeros((self.beta_0.shape[0], 1)), self.beta_0))

        # store dimensions
        self.M = self.YY.shape[0]
        self.K = self.YY.shape[1]

        # compute the total number of polymorphisms in each mutation rate class
        self.nn = np.sum(self.YY, axis=1)

        # initialize the empty latent SFS inferred from traditional optimization
        self.alpha_optim = None
        # initialize the fit probabilities inferred from traditional optimization
        self.fit_probs_optim = None

    def neutral_cdf(self, reverse = True):
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

        # if reverse, 0 bin equals 1
        # if not reverse most common bin equals 1
        
        neutral_sfs_copy = np.copy(self.neutral_sfs)
        bin_num = self.K

        if reverse:
            for i in range(1, bin_num):
                neutral_sfs_copy[:,:i] += einops.repeat(neutral_sfs_copy[:,i], 'h -> h repeat', repeat= i)
        else:
            for i in reversed(range(bin_num)):
                neutral_sfs_copy[:,i+1:] += einops.repeat(neutral_sfs_copy[:,i], 'h -> h repeat', repeat= bin_num - i - 1)
        
        return neutral_sfs_copy

    def _loglik(self, alpha, YY_samp=None):
        """
        Compute the log-likelihood of the observed polymorphism data given
        the non-decreasing latent SFS.
        
        Parameters
        ----------
        alpha : array_like
            The non-decreasing latent SFS.
            (K-1) array.
        
        Returns
        -------
        loglik : float
            The log-likelihood of the observed polymorphism data.
        """
        assert len(alpha) == self.K-1

        if YY_samp is None:
            YY_samp = self.YY
            nn_samp = self.nn
        else:
            assert YY_samp.shape == self.YY.shape
            nn_samp = np.sum(YY_samp, axis=1)
        # convert alpha to multinomial coefficients
        beta = np.cumsum(alpha)
        ZZ = np.sum(np.exp(self.beta_0 - beta), 1)
        loglik = np.sum(-nn_samp * np.log1p(ZZ) + np.sum((self.beta_0 - beta) * YY_samp[:, 1:], 1))
        # return the log-likelihood
        return -loglik
    
    def _gradient(self, alpha, YY_samp=None):
        """
        Compute the gradient of the log-likelihood of the observed polymorphism
        data given the non-decreasing latent SFS.
        
        Parameters
        ----------
        alpha : array_like
            The non-decreasing latent SFS.
            (K-1) array.
        
        Returns
        -------
        gradient : array_like
            The gradient of the log-likelihood of the observed polymorphism data.
            (K-1) array.
        """
        assert len(alpha) == self.K-1

        if YY_samp is None:
            YY_samp = self.YY
            nn_samp = self.nn
        else:
            assert YY_samp.shape == self.YY.shape
            nn_samp = np.sum(YY_samp, axis=1)
        # convert alpha to multinomial coefficients
        beta = np.cumsum(alpha)
        # compute the gradient
        ZZ = np.sum(np.exp(self.beta_0 - beta), 1)
        gradient = -np.sum(np.cumsum(YY_samp[:, 1:][:,::-1], 1)[:,::-1], 0) \
            + np.sum( nn_samp[:,None]*
                      np.cumsum(np.exp(self.beta_0 - beta)[:,::-1], 1)[:,::-1] / 
                      (1 + ZZ[:, None]), 0)
        return -gradient

    def gnocchi(self):
        """
        Compute the gnocchi estimate
        
        Parameters
        ----------
        
        Returns
        -------
        zscore : float
            The z-score of gnocchi.
        """
        #binary sfs for gnocchi
        assert self.neutral_sfs.shape[1] == 2

        polymorphic_p = np.copy(self.neutral_sfs)
        polymorphic_p[:,0] = 0
        return -1 * self.zscore(polymorphic_p)

    def extrainsight(self):
        """
        Compute the ExtRaINSIGHT estimate from dukler et al
        
        Parameters
        ----------
        
        Returns
        -------
        zscore : float
            The z-score of gnocchi.
        """
        #binary sfs for extrainsight
        assert self.neutral_sfs.shape[1] == 2

        monomorphic_p = np.copy(self.neutral_sfs)
        monomorphic_p[:,1] = 0
        
        return self.zscore(monomorphic_p)

    def zscore_cdf(self, transformation = "log", reverse=True):
        """
        Compute the z score of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        
        Returns
        -------
        zscore : float
            The z-score of cdf per-site.
        """

        neutral_dist = self.neutral_cdf(reverse)
        
        if transformation == "log":
            transformed_dist = np.log(neutral_dist)
        elif transformation == "none":
            transformed_dist = neutral_dist
        else:
            print("transformation is not included")

        return self.zscore(transformed_dist)
    
    def zscore(self, distribution):
        """
        Compute the z score of some transformation of distribution from neutral expectation.
        
        Parameters
        ----------
        
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

        return (data - neutral_expected_sum)/np.sqrt(neutral_var_sum)

    def ml_optim(self, jac=False, beta_max=100, verbose=True):
        """
        Perform maximum likelihood inference of the non-decreasing latent SFS
        using traditional optimization.

        Parameters
        ----------
        jac : bool
            Whether to use the gradient of the log-likelihood function.
        beta_max : float
            The maximum value of the inferred latent SFS.
        verbose : bool
            Whether to print the optimization results.

        Returns
        -------
        alpha_optim : array_like
            The inferred non-decreasing latent SFS.
            (K-1) array.
        fit_probs_optim : array_like
            The fit probabilities inferred from the inferred latent SFS.
            (M x K) array.
        """
        # initialize the optimization
        alpha0 = np.zeros(self.K-1)
        bounds = [(0, beta_max) for k in range(self.K-1)]
        # perform the optimization
        if not jac:
            res = opt.minimize(self._loglik, alpha0, bounds=bounds, method='L-BFGS-B')
        else:
            res = opt.minimize(self._loglik, alpha0, jac=self._gradient, 
                               bounds=bounds, method='L-BFGS-B')
        # store the inferred latent SFS
        if verbose:
            print(res)
        self.alpha_optim = res.x
        # convert the inferred latent SFS to multinomial coefficients and append zeros at the front
        beta = np.cumsum(self.alpha_optim)
        # compute the fit probabilities
        ZZ = np.sum(np.exp(self.beta_0 - beta), 1)
        self.fit_probs_optim = np.exp(self.beta_0_0 - np.hstack((np.zeros(1), beta))) / (1 + ZZ[:,None])

    def ml_boostrap(self, n_bs=100, verbose=True, beta_max=100):
        """
        Generate bootstrap samples of the observed polymorphism data and
        perform maximum likelihood inference of the non-decreasing latent SFS

        Parameters
        ----------
        n_bs : int
            The number of bootstrap samples to generate.

        Returns
        -------
        alpha_bs : array_like
            The inferred non-decreasing latent SFS for each bootstrap sample.
            (n_bs x K-1) array.
        """
        
        # initialize the array to store the inferred latent SFS for each bootstrap sample
        alpha_bs = np.zeros((n_bs, self.K-1))
        # initialize the optimization
        alpha0 = np.zeros(self.K-1)
        bounds = [(0, beta_max) for k in range(self.K-1)]
        # loop over bootstrap samples
        for i in range(n_bs):
            # generate a bootstrap sample of the observed polymorphism data
            YY_probs = self.YY.flatten() / np.sum(self.YY)
            YY_bs = np.random.multinomial(np.sum(self.YY), YY_probs).reshape(self.YY.shape)
            res = opt.minimize(self._loglik, alpha0, jac=self._gradient, 
                               bounds=bounds, method='L-BFGS-B', 
                               args=(YY_bs))
            # store the inferred latent SFS
            if verbose:
                print(res)
            alpha_bs[i,:] = res.x
        # compute the fit probabilities for each bootstrap sample
        beta_bs = np.cumsum(alpha_bs, axis=1)
        ZZ_bs = np.sum(np.exp(self.beta_0 - beta_bs[:,None,:]), -1)
        fit_probs_bs = np.exp(self.beta_0_0 - np.hstack((np.zeros((n_bs, 1)), beta_bs))[:,None,:]) / (1 + ZZ_bs[:,:,None])
        # return the inferred latent SFS for each bootstrap sample
        return alpha_bs, fit_probs_bs
        
    def KL(self, mu_ind):
        """
        Compute the KL divergence between the inferred latent SFS and the
        SFS expected under neutrality given the offset at a particular mutation
        rate.

        Parameters
        ----------
        mu_ind : int
            The index of the mutation rate class to use as a reference.

        Returns
        -------
        KL : float
        """
        # verify that the mutation rate index is valid
        assert mu_ind >= 0 and mu_ind < self.M
        # verify that the latent SFS has been inferred
        assert self.alpha_optim is not None
        # compute the KL divergence
        fit_sfs = self.fit_probs_optim[mu_ind,:]
        neut_sfs = np.exp(self.beta_0_0[mu_ind,:]) / np.sum(np.exp(self.beta_0_0[mu_ind,:]))
        neut_sfs = neut_sfs[fit_sfs>0]
        fit_sfs = fit_sfs[fit_sfs>0]
        KL = np.sum(fit_sfs * (np.log(fit_sfs) - np.log(neut_sfs)))
        return KL

