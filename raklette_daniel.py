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

import torch.distributions.transforms as transforms

from tqdm import tqdm
import matplotlib

## Useful transformations
####################################################
pad = torch.nn.ConstantPad1d((1,0), 0.)            # Add a 0 to a tensor
softmax = torch.nn.Softmax(-1)                     # softmax transform along the last dimension
relu = torch.nn.ReLU()                             # map everything < 0 -> 0 
order_trans = dist.transforms.OrderedTransform()   # y_0=x_0; y_i=y_0+sum_j=1^i exp(x_j) [not really used anymore, weird properties]
####################################################

## Helper functions for working with SFS simulations
####################################################
def make_bins(jmin, bb, nn, incl_zero=False):
    bins = np.arange(1, jmin + 1, dtype=int)
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
    result = np.zeros(len(bins)-1)
    for ii in range(1, len(bins)):
        result[ii-1] = np.mean(np.arange(bins[ii-1], bins[ii]))
    return result

def bin_sizes(bins):
    result = np.zeros(len(bins)-1)
    for ii in range(1, len(bins)):
        result[ii-1] = bins[ii] - bins[ii-1]
    return result

def bin_data(ac, nn, bins):
    result = np.zeros(len(bins)-1)
    for ii in range(1, len(bins)):
        sfs_entries = ((ac >= bins[ii-1]) & (ac < bins[ii]))
        result[ii-1] = np.sum(nn[sfs_entries])
    return result

def sfs_setup(data_path, suffix="set"):
    sfs_set = np.load(os.path.join(data_path, "sfs_{}.npy".format(suffix)))
    s_set = np.load(os.path.join(data_path, "s_{}.npy".format(suffix)))
    mu_set = np.load(os.path.join(data_path, "mu_{}.npy".format(suffix)))
    mac_set = np.load(os.path.join(data_path, "mac_{}.npy".format(suffix)))
    mu_vals = np.sort(np.unique(mu_set))
    s_vals = np.sort(np.unique(s_set))
    assert len(s_set) == len(mu_vals)*len(s_vals)
    sfs_reorder = np.zeros((len(mu_vals), len(s_vals), len(mac_set)))
    for ii, mu in enumerate(mu_vals):
        for jj, ss in enumerate(s_vals):
            sfs_ind = np.where((s_set==ss) & (mu_set==mu))[0]
            sfs_reorder[ii, jj, :] = sfs_set[:, sfs_ind].flatten()
    return sfs_reorder, mu_vals, s_vals, mac_set

def bin_sfs_setup(jmin, bb, data_path, suffix="set"):
    sfs, mu_vals, s_vals, mac_set = sfs_setup(data_path, suffix)
    sfs_bins = make_bins(jmin, bb, np.max(mac_set), incl_zero=True)
    sfs_bin_means = bin_means(sfs_bins)
    sfs_binned = np.zeros((len(mu_vals), len(s_vals), len(sfs_bin_means)))
    for ii, mu in enumerate(mu_vals):
        for jj, ss in enumerate(s_vals):
            sfs_binned[ii, jj, :] = bin_data(mac_set, sfs[ii, jj, :], sfs_bins)
            sfs_binned[ii, jj, :] /= np.sum(sfs_binned[ii, jj, :])
    return {"sfs_binned":sfs_binned, "mu_vals":mu_vals, "s_vals":s_vals, "sfs_bin_means":sfs_bin_means, "sfs_bins":sfs_bins}
####################################################

## Transformations of the SFS
####################################################
def multinomial_trans(sfs_probs, offset=None):
    sfs_probs = np.array(sfs_probs)
    P_0 = sfs_probs[...,0]
    if offset:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None]) - offset
    else:
        betas = np.log(sfs_probs[...,1:]) - np.log(P_0[...,None])
    return betas

def multinomial_trans_torch(sfs_probs):
    P_0 = sfs_probs[...,0]
    return torch.log(sfs_probs[...,1:]) - torch.log(P_0[...,None])

def KL_fw(neut_probs, sel_probs):
    return np.sum(neut_probs * (np.log(neut_probs) - np.log(sel_probs)), axis=-1)

def KL_rv(neut_probs, sel_probs):
    return np.sum(sel_probs * (np.log(sel_probs) - np.log(neut_probs)), axis=-1)
####################################################

## Simulate SFS data
####################################################
def simulate_mu_sfs_gene_sample(mu_alpha, dfe_alpha, sfs_mat, n_genes, gene_sizes, covariate_effects=None):
    mu_alpha  = pyro.param("mu_alpha",  mu_alpha,  constraint=constraints.positive).type(torch.float32)
    dfe_alpha = pyro.param("dfe_alpha", dfe_alpha.type(torch.float32), constraint=constraints.positive).type(torch.float32)
    with pyro.plate("genes", n_genes):
        dfe = pyro.sample("dfe", dist.Dirichlet(dfe_alpha))
        mu_dist = pyro.sample("mu", dist.Dirichlet(mu_alpha))
    sfs = torch.matmul(dfe, sfs_mat)
    mu_vals = torch.concat([dist.Categorical(mu_dist[ii,:]).sample((gene_size,)) for ii, gene_size in enumerate(gene_sizes)])
    gene_ids = torch.concat([torch.tensor(ii).repeat(gene_size) for ii, gene_size in enumerate(gene_sizes)])
    assert mu_vals.shape == gene_ids.shape
    if covariate_effects is not None:
        sfs_beta = multinomial_trans_torch(sfs)
        samps = dist.Categorical(softmax(pad(sfs_beta[mu_vals, gene_ids, :] + covariate_effects))).sample()
    else:
        samps = dist.Categorical(sfs[mu_vals,gene_ids,:]).sample()
    return {"samps":samps, "mu_vals":mu_vals, "gene_ids":gene_ids, "dfe":dfe, "mu_dist":mu_dist}
####################################################

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

def calc_KL_genewise(fit, ref_mu_ii=None):
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

    beta_prior_b = fit["beta_prior_b"] # need the interaction term between mutation rate and selection
    post_samples = fit["post_samples"] # grab dictionary with posterior samples

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
    
    # transform to probabilities (SFS) for ecah gene and store as a numpy array
    post_probs = softmax(pad(beta_neut - post_trans -
                              mu_ref[ref_mu_ii]*torch.cumsum(beta_prior_b, -1)*post_trans
                             )
                         ).detach().numpy()
    
    # calculate KL for each draw from the posterior distribution for each gene
    KL_fw_post = KL_fw(neut_sfs_full[ref_mu_ii,:].detach().numpy(), post_probs)
    KL_rv_post = KL_rv(neut_sfs_full[ref_mu_ii,:].detach().numpy(), post_probs)
    
    fit["post_probs"] = post_probs
    fit["KL_fw_post"] = KL_fw_post
    fit["KL_rv_post"] = KL_rv_post
    
    return fit

def mu_sfs_sitewise_regr_cov(beta_neut, mu_vals, gene_ids, covariates, n_bins, mu_ref, 
                             sample_sfs=None, n_mix=2, cov_sigma_prior=torch.tensor(0.1, dtype=torch.float32), trans="abs", pdist="t"):
    """
    Pyro sampling model for a gene-based DFE with covariates
    """
    n_covs = covariates.shape[-1]          # number of covariates included
    n_sites = len(mu_vals)                 # number of sites (potential mutations) we are modeling
    n_genes = len(torch.unique(gene_ids))  # number of genes
    mu = torch.unique(mu_vals)             # set of all possible mutation rates
    n_mu = len(mu)                         # number of unique mutation rates
    
    ## Setup flexible prior
    # parameters describing the prior over genes are set as pyro.param, meaning they will get point estimates (no posterior)
    if pdist=="t":
        # t-distribution can modulate covariance (L) and kurtosis (df)
        # uses a fixed "point mass" at zero as one of the mixtures, not sure if this should be kept
        beta_prior_mean = pyro.param("beta_prior_mean", torch.randn((n_mix-1,n_bins)),
                                     constraint=constraints.real)
        beta_prior_L = pyro.param("beta_prior_L", torch.linalg.cholesky(0.01*torch.diag(torch.ones(n_bins, dtype=torch.float32))).expand(n_mix-1, n_bins, n_bins), 
                                                                        constraint=constraints.lower_cholesky)
        beta_prior_df = pyro.param("beta_prior_df", torch.tensor([10]*(n_mix-1), dtype=torch.float32), constraint=constraints.positive)
        mix_probs = pyro.param("mix_probs", torch.ones(n_mix, dtype=torch.float32)/n_mix, constraint=constraints.simplex)
    elif pdist=="normal":
        # normal model has zero covariance, a different variance for each bin though
        mix_probs = pyro.param("mix_probs", torch.ones(n_mix, dtype=torch.float32)/n_mix, constraint=constraints.simplex)
        beta_prior_loc = pyro.param("beta_prior_loc", torch.randn((n_mix, n_bins)), constraint=constraints.real)
        beta_prior_scale = pyro.param("beta_prior_scale", torch.rand((n_mix, n_bins)), constraint=constraints.positive)
        
    # interaction term bewteen gene-based selection and mutation rate
    beta_prior_b = pyro.param("beta_prior_b", torch.tensor([0.001]*n_bins, dtype=torch.float32), constraint=constraints.positive)
    
    # Each covariate has a vector of betas, one for each bin, maybe think about different prior here?
    with pyro.plate("covariates", n_covs):
        beta_cov = pyro.sample("beta_cov", dist.HalfCauchy(cov_sigma_prior).expand([n_bins]).to_event(1))
    
    with pyro.plate("genes", n_genes):
        # sample latent betas from either t or normal distribution
        if pdist=="t":
            beta_sel = pyro.sample("beta_sel", dist.MixtureSameFamily(dist.Categorical(mix_probs),
                                   dist.MultivariateStudentT(df=torch.cat((beta_prior_df, torch.tensor([1000], dtype=torch.float32))), 
                                                             loc=torch.cat((beta_prior_mean, 
                                                                            torch.tensor([0]*n_bins, dtype=torch.float32).expand((1, n_bins)))), 
                                                             scale_tril=torch.cat((beta_prior_L, 
                                                                                   torch.linalg.cholesky(torch.diag(1e-8*torch.ones(n_bins, dtype=torch.float32))).expand(1, n_bins, n_bins))))))
        elif pdist=="normal":
            beta_sel = pyro.sample("beta_sel", dist.MixtureSameFamily(dist.Categorical(mix_probs),
                                                                      dist.Normal(beta_prior_loc, beta_prior_scale).to_event(1)))
        # apply transform to latent betas
        if trans == "abs":
            beta_trans = torch.cumsum(torch.abs(beta_sel), dim=-1)
        elif trans=="logabs":
            beta_trans = torch.cumsum(torch.log(torch.abs(beta_sel)+1), dim=-1)
        elif trans=="relu":
            beta_trans = torch.cumsum(relu(beta_sel), dim=-1)
        elif trans=="logrelu":
            beta_trans = torch.cumsum(torch.log(relu(beta_sel)+1), dim=-1)
        
    # calculate the multinomial coefficients for each gene and each mutation rate
    mu_adj = mu_ref[...,None] * torch.cumsum(beta_prior_b, -1) * beta_trans[...,None,:]
    mn_sfs = (beta_neut  - 
              beta_trans[...,None,:] -
              mu_adj)
    # convert to probabilities per-site and adjust for covariates
    sfs = softmax(pad(mn_sfs[..., gene_ids, mu_vals, :] - torch.matmul(covariates, torch.cumsum(beta_cov, -1))))
    
    with pyro.plate("sites", n_sites):
        pyro.sample("obs", dist.Categorical(sfs), obs=sample_sfs)
        
def fit_mu_sfs_sitewise_regr_cov(neut_sfs_full, mu_vals, gene_ids, covariates, n_bins, sample_sfs, mu_ref,
                                 lr=0.005, num_particles=1, n_steps=1000, post_samps=10000, ref_mu_ii=-1, n_mix=2, trans="abs", pdist="t"):
    """
    Fit a model with gene-based DFE as well as covariates and calculate posterior predictions
    """
    beta_neut_full = multinomial_trans_torch(neut_sfs_full)
    # use the mean-field normal guide, need to include options for other guides
    guide = pyro.infer.autoguide.AutoNormal(mu_sfs_sitewise_regr_cov)
    pyro.clear_param_store()
    
    # run SVI
    adam = pyro.optim.Adam({"lr":lr})
    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True)
    svi = pyro.infer.SVI(mu_sfs_sitewise_regr_cov, guide, adam, elbo)
    losses = []
    for step in tqdm(range(n_steps)): # tqdm is just a progress bar thing 
        loss = svi.step(beta_neut_full, mu_vals, gene_ids, covariates, n_bins, mu_ref, 
                        sample_sfs, n_mix, torch.tensor(0.1, dtype=torch.float32), trans, pdist)
        losses.append(loss)
    fig, ax = plot_losses(losses)
    
    # grab gene-DFE prior parameter point estimates
    beta_neut = beta_neut_full[ref_mu_ii,:]
    if pdist=="t":
        beta_prior_df = pyro.param("beta_prior_df")
        beta_prior_mean = pyro.param("beta_prior_mean")
        beta_prior_L = pyro.param("beta_prior_L")
        mix_probs = pyro.param("mix_probs")
    elif pdist=="normal":
        mix_probs = pyro.param("mix_probs")
        beta_prior_loc = pyro.param("beta_prior_loc")
        beta_prior_scale = pyro.param("beta_prior_scale")
        
    beta_prior_b = pyro.param("beta_prior_b")

    # Sample betas from the DFE prior, representing the fit distribution across genes
    if pdist=="t":
        prior_dist = dist.MixtureSameFamily(dist.Categorical(mix_probs),
                                              dist.MultivariateStudentT(df=torch.cat((beta_prior_df, torch.tensor([1000], dtype=float))), 
                                                                    loc=torch.cat((beta_prior_mean, 
                                                                                       torch.tensor([0]*n_bins, dtype=float).expand((1, n_bins)))), 
                                                                    scale_tril=torch.cat((beta_prior_L, 
                                                                                torch.linalg.cholesky(torch.diag(1e-8*torch.ones(n_bins, dtype=float))).expand(1, n_bins, n_bins))))
                                                                 )
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

    ## Prior SFS probabilities for gene effects in the absence of covariates
    prior_probs = softmax(pad(beta_neut - prior_trans -
                              mu_ref[ref_mu_ii]*torch.cumsum(beta_prior_b, -1)*prior_trans
                             )
                         ).detach().numpy()
    
    # take samples from the posterior distribution on all betas
    with pyro.plate("samples", post_samps, dim=-2):
        post_samples = guide()
    
    if pdist=="t":
        result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                  "beta_prior_df":beta_prior_df, "beta_prior_mean":beta_prior_mean, "beta_prior_L":beta_prior_L,
                  "mix_probs":mix_probs, 
                  "beta_prior_b":beta_prior_b, "trans":trans,
                  "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref}
    elif pdist=="normal":
        result = {"neut_sfs_full":neut_sfs_full, "beta_neut_full":beta_neut_full, "ref_mu_ii":ref_mu_ii,
                  "beta_prior_scale":beta_prior_scale, "beta_prior_loc":beta_prior_loc,
                  "mix_probs":mix_probs,
                  "beta_prior_b":beta_prior_b, "trans":trans,
                  "prior_probs":prior_probs, "post_samples":post_samples, "mu_ref":mu_ref}
        
    # calculate the posterior distribution on KL for each gene
    result = calc_KL_genewise(result, pdist=pdist)
    
    ## Then calculate the posteriors for covariate betas
    result["post_beta_cov"] = torch.cumsum(post_samples['beta_cov'], -1)
    result["losses"] = losses
    result["fig"] = (fig, ax)
    
    return result
####################################################