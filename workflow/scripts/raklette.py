import logging
import os
import math

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule, PyroParam, PyroSample

import torch.distributions.transforms as transforms

from tqdm import tqdm
import matplotlib

## Useful transformations
####################################################
pad = torch.nn.ConstantPad1d((1,0), 0.)           # Add a 0 to a tensor
softmax = torch.nn.Softmax(-1)                    # softmax transform along the last dimension
relu = torch.nn.ReLU()                            # map everything < 0 -> 0
order_trans = dist.transforms.OrderedTransform()  # y_0=x_0; y_i=y_0+sum_j=1^i exp(x_j) [not really used anymore, weird properties]
####################################################

## Transformations of the SFS
####################################################
def multinomial_trans(sfs_probs, offset=None):
    """
    Transforms the probabilities of a site frequency spectrum (SFS) into log-odds ratios.
    
    Parameters:
    sfs_probs (array-like): The probabilities of the SFS.
    offset (float or None): The offset value to be subtracted from the log-odds ratios. Default is None.
    
    Returns:
    log_odds (array-like): The log-odds ratios of the SFS probabilities.
    """

    sfs_probs = np.array(sfs_probs)
    P_0 = sfs_probs[...,0]
    if offset is not None:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None]) - offset
    else:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None])
    return betas

def multinomial_trans_torch(sfs_probs):
    P_0 = sfs_probs[...,0]
    return torch.log(sfs_probs[...,1:]) - torch.log(P_0[...,None])

def KL_fw(neut_probs, sel_probs):
    """
    Calculates the Kullback-Leibler divergence between two probability distributions.
    
    Parameters:
        neut_probs (numpy.ndarray): The probability distribution of the neutral model.
        sel_probs (numpy.ndarray): The probability distribution of the selection model.
    
    Returns:
        numpy.ndarray: The Kullback-Leibler divergence between the two probability distributions.
    """
    return np.sum(neut_probs * (np.log(neut_probs) - np.log(sel_probs)), axis=-1)

def KL_rv(neut_probs, sel_probs):
    return np.sum(sel_probs * (np.log(sel_probs) - np.log(neut_probs)), axis=-1)


# function to 

## Model fitting
####################################################
def plot_losses(losses):
    """
    Make a plot of SVI losses
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("SVI step")
    ax.set_ylabel("ELBO loss");
    ax.set_yscale("log")
    return fig, ax

def calc_KL_genewise(fit, fit_interaction, ref_mu_ii=None):
    """
    Calculate KL between the neutral distribution at a reference mutation rate and 
    the fit selected distribution at each gene.
    """
    neut_sfs_full = fit["neut_sfs_full"]
    beta_neut_full = fit["beta_neut_full"]
    mu_ref = fit["mu_ref"]
    if ref_mu_ii:
        fit["ref_mu_ii"] = ref_mu_ii
    ref_mu_ii = fit["ref_mu_ii"]
    beta_neut = beta_neut_full[ref_mu_ii,:] # neutral betas for the reference mutation rate
    
    if fit_interaction:
        beta_prior_b = fit["beta_prior_b"] # need the interaction term between mutation rate and selection
    post_samples = fit["post_samples"] # grab dictionary with posterior samples

    if fit["fit_prior"]:    
        # apply the correction transormation to the latent beta_sel
        # (definitely the kind of thing to store elsewhere in the class structure)
        if fit["trans"] == "abs":
            post_trans = torch.cumsum(torch.abs(post_samples["beta_sel"]), dim=-1)
        elif fit["trans"] =="logabs":
            post_trans = torch.cumsum(torch.log(torch.abs(post_samples["beta_sel"])+1), dim=-1)
        elif fit["trans"]=="relu":
            post_trans = torch.cumsum(relu(post_samples["beta_sel"]), dim=-1)
        elif fit["trans"]=="logrelu":
            post_trans = torch.cumsum(torch.log(relu(post_samples["beta_sel"])+1), dim=-1)
    else:
         post_trans = torch.cumsum(post_samples["beta_sel"], dim=-1)
    
    # transform to probabilities (SFS) for ecah gene and store as a numpy array
    if fit_interaction:
        post_probs = softmax(pad(beta_neut - post_trans -
                              mu_ref[ref_mu_ii]*torch.cumsum(beta_prior_b, -1)*post_trans)).detach().numpy()
    else:
        post_probs = softmax(pad(beta_neut - post_trans)).detach().numpy()
    
    # calculate KL for each draw from the posterior distribution for each gene
    KL_fw_post = KL_fw(neut_sfs_full[ref_mu_ii,:].detach().numpy(), post_probs)
    KL_rv_post = KL_rv(neut_sfs_full[ref_mu_ii,:].detach().numpy(), post_probs)
    
    fit["post_probs"] = post_probs
    fit["KL_fw_post"] = KL_fw_post
    fit["KL_rv_post"] = KL_rv_post
    
    return fit

###################################################raklette running using neural layers########################################
class Linear_first(PyroModule):
    def __init__(self, in_size, out_size, cov_sigma_prior):
        super().__init__()
        self.weight = PyroSample(dist.HalfCauchy(cov_sigma_prior).expand([in_size, out_size]).to_event(1))
        
    def forward(self, input_):
        return input_ @ self.weight
    

class softmax_sfs(PyroModule):
    def __init__(self, beta_neut_full):
        super().__init__()
        # calculate the multinomial coefficients for each mutation rate
        self.mn_sfs = beta_neut_full[None,...]

    def forward(self, mu_vals, input_):
        
        #I want to check if the cumulative sum is still correct
        input_cumsum = torch.cumsum(input_, dim=-1)
        # convert to probabilities per-site and adjust for covariates
        sfs = softmax(pad(self.mn_sfs[..., mu_vals, :] - input_cumsum))
        
        return sfs

class raklette_neuralnet():
    def __init__(self, neut_sfs_full, n_bins, n_covs, cov_sigma_prior = torch.tensor(0.1, dtype=torch.float32),
                 ref_mu_ii=-1):

        beta_neut_full = multinomial_trans_torch(neut_sfs_full) #neut_sfs_full is the neutral sfs
        
        self.fc1 = Linear_first(n_covs, n_bins, cov_sigma_prior)  # set up first FC layer
        self.softmax_sfs = softmax_sfs(beta_neut_full)
        

    def model(self, mu_vals, covariates, sample_sfs=None):
        '''
        Run the model for a given dataset, defined by mutation rates, gene_ids, and other covariates

        Parameters
        ----------
        mu_vals :
            The mutation rate for each site being analyzed by the model as a 1D array, proportional to per-generation ideally
        covariates :
            Matrix of covariates
        sample_sfs :
            The observed binned SFS data we are fitting the model to
        '''
        n_sites = len(mu_vals)
            
        ly1 = self.fc1(covariates)
        sfs = self.softmax_sfs(mu_vals, ly1)
        
        with pyro.plate("sites", n_sites):
            pyro.sample("obs", dist.Categorical(sfs), obs=sample_sfs)

###################################################raklette running using covariates########################################
class raklette_cov():
    def __init__(self, neut_sfs_full, n_bins, n_covs, cov_sigma_prior = torch.tensor(0.1, dtype=torch.float32),
                 ref_mu_ii=-1):

#         mu = torch.unique(mu_vals)             # set of all possible mutation rates
#         n_mu = len(mu)                         # number of unique mutation rates

        beta_neut_full = multinomial_trans_torch(neut_sfs_full) #neut_sfs_full is the neutral sfs
        self.beta_neut_full = beta_neut_full

        beta_neut = beta_neut_full[ref_mu_ii,:]
        self.beta_neut = beta_neut

        self.n_bins = n_bins
        self.n_covs = n_covs

        self.cov_sigma_prior = cov_sigma_prior

    def model(self, mu_vals, covariates, sample_sfs=None):
        '''
        Run the model for a given dataset, defined by mutation rates, gene_ids, and other covariates

        Parameters
        ----------
        mu_vals :
            The mutation rate for each site being analyzed by the model as a 1D array, proportional to per-generation ideally
        covariates :
            Matrix of covariates
        sample_sfs :
            The observed binned SFS data we are fitting the model to

        '''
        n_sites = len(mu_vals)
            
        #beta for covariates
        beta_cov = pyro.sample("beta_cov", dist.HalfCauchy(self.cov_sigma_prior).expand([self.n_covs, self.n_bins]).to_event(1))    
            
        beta_cov_trans = torch.cumsum(beta_cov, dim=-1)

        # calculate the multinomial coefficients for each mutation rate
        mn_sfs = self.beta_neut_full[None,...]
        
        # convert to probabilities per-site and adjust for covariates
        sfs = softmax(pad(mn_sfs[..., mu_vals, :] - torch.matmul(covariates, beta_cov_trans)))
        
        with pyro.plate("sites", n_sites):
            pyro.sample("obs", dist.Categorical(sfs), obs=sample_sfs)
            
def post_analysis_cov(neutral_sfs, mu_ref, n_bins, n_covs, losses, cov_sigma_prior, ref_mu_ii = -1):

    neut_sfs_full = neutral_sfs

    beta_neut_full = multinomial_trans_torch(neut_sfs_full) #neut_sfs_full is the neutral sfs
    # grab gene-DFE prior parameter point estimates
    beta_neut = beta_neut_full[ref_mu_ii,:]

    #make result into a dictionary
    result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii, "cov_sigma_prior":cov_sigma_prior}

#     result["post_beta_cov"] = torch.cumsum(post_samples['beta_cov'], -1)
    result["losses"] = losses

    fig, ax = plot_losses(losses)

    result["fig"] = (fig, ax)

    return result
            
################################################raklette running covariates + genes ########################################
class raklette():
    def __init__(self, neut_sfs_full, n_bins, mu_ref, n_covs, n_genes, cov_sigma_prior = torch.tensor(0.1, dtype=torch.float32),
                 n_mix=2, ref_mu_ii=-1,
                 trans="abs", pdist="t", fit_prior = True, fit_interaction = False):

#         mu = torch.unique(mu_vals)             # set of all possible mutation rates
#         n_mu = len(mu)                         # number of unique mutation rates

        beta_neut_full = multinomial_trans_torch(neut_sfs_full) #neut_sfs_full is the neutral sfs
        self.beta_neut_full = beta_neut_full

        beta_neut = beta_neut_full[ref_mu_ii,:]
        self.beta_neut = beta_neut

        self.n_bins = n_bins
        self.mu_ref = mu_ref
        self.n_covs = n_covs
        self.n_genes = n_genes

        self.n_mix = n_mix
        self.cov_sigma_prior = cov_sigma_prior
        self.trans = trans
        self.pdist = pdist
        self.prior = fit_prior
        self.interaction = fit_interaction

    def model(self, mu_vals, gene_ids, covariates = None, sample_sfs=None):
        '''
        Run the model for a given dataset, defined by mutation rates, gene_ids, and other covariates

        Parameters
        ----------
        mu_vals :
            The mutation rate for each site being analyzed by the model as a 1D array, proportional to per-generation ideally
        gene_ids :
            The gene id for each site being analyzed, categorical variable, give as integers
        covariates :
            Matrix of covariates
        sample_sfs :
            The observed binned SFS data we are fitting the model to

        '''

        n_sites = len(mu_vals)
        
        #if we want to fit a specific prior to genes
        if self.prior:
            ## Setup flexible prior
            # parameters describing the prior over genes are set as pyro.param, meaning they will get point estimates (no posterior)
            if self.pdist=="t":
                # t-distribution can modulate covariance (L) and kurtosis (df)
                # uses a fixed "point mass" at zero as one of the mixtures, not sure if this should be kept
                beta_prior_mean = pyro.param("beta_prior_mean", torch.randn((self.n_mix-1,self.n_bins)),
                                             constraint=constraints.real)
                beta_prior_L = pyro.param("beta_prior_L", torch.linalg.cholesky(0.01*torch.diag(torch.ones(self.n_bins, dtype=torch.float32))).expand(self.n_mix-1, self.n_bins, self.n_bins), constraint=constraints.lower_cholesky)
                beta_prior_df = pyro.param("beta_prior_df", torch.tensor([10]*(self.n_mix-1), dtype=torch.float32), constraint=constraints.positive)
                mix_probs = pyro.param("mix_probs", torch.ones(self.n_mix, dtype=torch.float32)/self.n_mix, constraint=constraints.simplex)
            elif self.pdist=="normal":
                # normal model has zero covariance, a different variance for each bin though
                mix_probs = pyro.param("mix_probs", torch.ones(self.n_mix, dtype=torch.float32)/self.n_mix, constraint=constraints.simplex)
                beta_prior_loc = pyro.param("beta_prior_loc", torch.randn((self.n_mix, self.n_bins)), constraint=constraints.real)
                beta_prior_scale = pyro.param("beta_prior_scale", torch.rand((self.n_mix, self.n_bins)), constraint=constraints.positive)

            with pyro.plate("genes", self.n_genes):
                # sample latent betas from either t or normal distribution
                if self.pdist=="t":
                    beta_sel = pyro.sample("beta_sel", dist.MixtureSameFamily(dist.Categorical(mix_probs),
                                           dist.MultivariateStudentT(df=torch.cat((beta_prior_df, torch.tensor([1000], dtype=torch.float32))), loc=torch.cat((beta_prior_mean, torch.tensor([0]*self.n_bins, dtype=torch.float32).expand((1, self.n_bins)))), scale_tril=torch.cat((beta_prior_L, torch.linalg.cholesky(torch.diag(1e-8*torch.ones(self.n_bins, dtype=torch.float32))).expand(1, self.n_bins, self.n_bins))))))
                elif self.pdist=="normal":
                    beta_sel = pyro.sample("beta_sel", dist.MixtureSameFamily(dist.Categorical(mix_probs), dist.Normal(beta_prior_loc, beta_prior_scale).to_event(1)))

                # apply transform to latent betas
                if self.trans == "abs":
                    beta_trans = torch.cumsum(torch.abs(beta_sel), dim=-1)
                elif self.trans=="logabs":
                    beta_trans = torch.cumsum(torch.log(torch.abs(beta_sel)+1), dim=-1)
                elif self.trans=="relu":
                    beta_trans = torch.cumsum(relu(beta_sel), dim=-1)
                elif self.trans=="logrelu":
                    beta_trans = torch.cumsum(torch.log(relu(beta_sel)+1), dim=-1)

        else:
            with pyro.plate("genes", self.n_genes):
                beta_sel = pyro.sample("beta_sel", dist.HalfCauchy(self.cov_sigma_prior).expand([self.n_bins]).to_event(1))
            
            beta_trans = torch.cumsum(beta_sel, dim=-1)
            
        # interaction term bewteen gene-based selection and mutation rate
        if self.interaction:
            beta_prior_b = pyro.param("beta_prior_b", torch.tensor([0.001]*self.n_bins, dtype=torch.float32), constraint=constraints.positive)
            
        #beta for covariates
        if self.n_covs > 0:
            # Each covariate has a vector of betas, one for each bin, maybe think about different prior here?
            with pyro.plate("covariates", self.n_covs):
                beta_cov = pyro.sample("beta_cov", dist.HalfCauchy(self.cov_sigma_prior).expand([self.n_bins]).to_event(1))

        # calculate the multinomial coefficients for each gene and each mutation rate
        if self.interaction:
            mu_adj = self.mu_ref[...,None] * torch.cumsum(beta_prior_b, -1) * beta_trans[...,None,:]
            
            mn_sfs = (self.beta_neut_full[None,...] -
                  beta_trans[...,None,:] -
                  mu_adj)
        else:
            mn_sfs = self.beta_neut_full[None,...] - beta_trans[...,None,:]
        
        # convert to probabilities per-site and adjust for covariates
        if self.n_covs > 0:
            sfs = softmax(pad(mn_sfs[..., gene_ids, mu_vals, :] - covariates * torch.cumsum(beta_cov, -1)))
        else:
            sfs = softmax(pad(mn_sfs[..., gene_ids, mu_vals, :]))

        with pyro.plate("sites", n_sites):
            pyro.sample("obs", dist.Categorical(sfs), obs=sample_sfs)
            
def post_analysis(neutral_sfs, mu_ref, n_bins, guide, n_covs, losses, cov_sigma_prior, ref_mu_ii = -1, pdist = "t", trans = "abs", post_samps=10000, fit_prior = True, fit_interaction = False):
#     bin_columns = []
#     for i in range(5):
#         bin_columns.append(str(i) + "_bin")
#     neutral_sfs = torch.tensor(sfs[bin_columns].values)
#     mu_ref = torch.tensor(sfs["mu"].values)
#     n_bins = len(neutral_sfs[1]) - 1
    
    neut_sfs_full = neutral_sfs
    
    beta_neut_full = multinomial_trans_torch(neut_sfs_full) #neut_sfs_full is the neutral sfs
    # grab gene-DFE prior parameter point estimates
    beta_neut = beta_neut_full[ref_mu_ii,:]
    
    if fit_prior:    
        if pdist=="t":
            beta_prior_df = pyro.param("beta_prior_df")
            beta_prior_mean = pyro.param("beta_prior_mean")
            beta_prior_L = pyro.param("beta_prior_L")
            mix_probs = pyro.param("mix_probs")
        elif pdist=="normal":
            mix_probs = pyro.param("mix_probs")
            beta_prior_loc = pyro.param("beta_prior_loc")
            beta_prior_scale = pyro.param("beta_prior_scale")
            
        # Sample betas from the DFE prior, representing the fit distribution across genes
        if pdist=="t":
            prior_dist = dist.MixtureSameFamily(dist.Categorical(mix_probs), dist.MultivariateStudentT(df=torch.cat((beta_prior_df, torch.tensor([1000], dtype=float))),loc=torch.cat((beta_prior_mean, torch.tensor([0]*n_bins, dtype=float).expand((1, n_bins)))), scale_tril=torch.cat((beta_prior_L, torch.linalg.cholesky(torch.diag(1e-8*torch.ones(n_bins, dtype=float))).expand(1, n_bins, n_bins)))))
        elif pdist=="normal":
            prior_dist = dist.MixtureSameFamily(dist.Categorical(mix_probs),
                                                dist.Normal(beta_prior_loc, beta_prior_scale).to_event(1))
                         
        prior_samps = prior_dist.sample((post_samps,))

        if trans == "abs":
            prior_trans = torch.cumsum(torch.abs(prior_samps), axis=-1)
        elif trans=="logabs":
            prior_trans = torch.cumsum(torch.log(torch.abs(prior_samps)+1), axis=-1)
        elif trans=="relu":
            prior_trans = torch.cumsum(relu(prior_samps), axis=-1)
        elif trans=="logrelu":
            prior_trans = torch.cumsum(torch.log(relu(prior_samps)+1), axis=-1)            
    else:
        prior_dist = dist.HalfCauchy(cov_sigma_prior).expand([n_bins]).to_event(1)
        
        prior_samps = prior_dist.sample((post_samps,))
        prior_trans = torch.cumsum(prior_samps, dim=-1)
        
    if fit_interaction:
        beta_prior_b = pyro.param("beta_prior_b")
        
        ## Prior SFS probabilities for gene effects in the absence of covariates
        prior_probs = softmax(pad(beta_neut - prior_trans -
                                  mu_ref[ref_mu_ii]*torch.cumsum(beta_prior_b, -1)*prior_trans)).detach().numpy()
        
    else:
        ## Prior SFS probabilities for gene effects in the absence of covariates
        prior_probs = softmax(pad(beta_neut - prior_trans)).detach().numpy()


    # take samples from the posterior distribution on all betas
    with pyro.plate("samples", post_samps, dim=-2):
        post_samples = guide()
        
        
    #make result into a dictionary
    if fit_prior:
        if pdist=="t":
            if fit_interaction:
                result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                          "beta_prior_df":beta_prior_df, "beta_prior_mean":beta_prior_mean, "beta_prior_L":beta_prior_L,
                          "mix_probs":mix_probs, 
                          "beta_prior_b":beta_prior_b, "trans":trans,
                          "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref, "fit_prior":fit_prior}
            else:
                result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                          "beta_prior_df":beta_prior_df, "beta_prior_mean":beta_prior_mean, "beta_prior_L":beta_prior_L,
                          "mix_probs":mix_probs, "trans":trans,
                          "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref, "fit_prior":fit_prior}
        elif pdist=="normal":
            if fit_interaction:
                result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                          "beta_prior_scale":beta_prior_scale, "beta_prior_loc":beta_prior_loc,
                          "mix_probs":mix_probs, "beta_prior_b":beta_prior_b, "trans":trans,
                          "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref, "fit_prior":fit_prior}
            else:
                result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                          "beta_prior_scale":beta_prior_scale, "beta_prior_loc":beta_prior_loc,
                          "mix_probs":mix_probs,"trans":trans,
                          "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref, "fit_prior":fit_prior}
    else:
        if fit_interaction:
            result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                      "beta_prior_b":beta_prior_b, "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref,
                      "fit_prior":fit_prior}
        else:
            result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                      "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref,
                      "fit_prior":fit_prior}

    # calculate the posterior distribution on KL for each gene
    result = calc_KL_genewise(result, fit_interaction)

    ## Then calculate the posteriors for covariate betas
    if n_covs > 0:
        result["post_beta_cov"] = torch.cumsum(post_samples['beta_cov'], -1)
    result["losses"] = losses
    
    fig, ax = plot_losses(losses)
    
    result["fig"] = (fig, ax)
    
    return result