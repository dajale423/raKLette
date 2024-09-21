import sys
import pandas as pd
import torch

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule

import datetime
import time

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL/scripts')

import raklette
import pickle

KL_data_dir = "/home/djl34/lab_pd/kl/data"

########################################################################################################
# Create Dataset
########################################################################################################
    
class TSVDataset(Dataset):
    def __init__(self, header, chunksize, chrom = None):
        # self.path = path
        self.chunksize = chunksize

        if chrom is None:
            filename_list = glob.glob(header + f"chr_*_chunk_{chunksize}_*.tsv")
        else:
            filename_list = glob.glob(header + f"chr_{chrom}_chunk_{chunksize}_*.tsv")

        self.len = len(filename_list)
        self.filename_list = filename_list
        
    def __getitem__(self, index):
        
        x = pd.read_csv(self.filename_list[index], sep = "\t")
        x = x.dropna()
        x = x.astype(float)
        x = torch.from_numpy(x.values)
        return x

    def __len__(self):
        return self.len


########################################################################################################
# Main Function
########################################################################################################


def run_raklette(inference, loader, post_analysis, n_covs, num_epochs, neutral_sfs_filename, output_filename, lr, gamma, cov_sigma_prior = torch.tensor(0.1, dtype=torch.float32), mu_col = 0, bin_col = 1, cov_col = 2, print_loss = True):
    #lr is initial learning rate

    print("running raklette", flush=True)

    # read neutral sfs
    sfs = pd.read_csv(neutral_sfs_filename, sep = "\t")
    bin_columns = []
    for i in range(len(sfs.columns) - 2):
        bin_columns.append(str(float(i)))
    neutral_sfs = torch.tensor(sfs[bin_columns].values)
    mu_ref = torch.tensor(sfs["mu"].values)
    n_bins = len(neutral_sfs[1]) - 1
    print("number of bins: " + str(n_bins), flush = True)

    KL = inference(neutral_sfs, n_bins, n_covs, cov_sigma_prior = cov_sigma_prior)
                
    model = KL.model
    guide = pyro.infer.autoguide.AutoNormal(model)

    #run inference
    pyro.clear_param_store()
    
    num_steps = num_epochs * len(loader)
    lrd = gamma ** (1 / num_steps)
    
    # run SVI
    optimizer = pyro.optim.ClippedAdam({"lr":lr, 'lrd': lrd})
#     optimizer = pyro.optim.Adam({"lr":lr})
    elbo = pyro.infer.Trace_ELBO(num_particles=1, vectorize_particles=True)
    svi = pyro.infer.SVI(model, guide, optimizer, elbo)
    losses = []
    
    start = time.time()
    divide_by = 1

    for epoch in range(num_epochs):
        if print_loss:
            print("epoch: " + str(epoch), flush = True)

        # Take a gradient step for each mini-batch in the dataset
        for batch_idx, data in enumerate(loader):
            if batch_idx % divide_by == 0:
                if print_loss:
                    print("\t batch: " + str(batch_idx), flush = True)
                end = time.time()
                if print_loss:
                    print("\t\t time from last batch: " + str(end - start), flush = True)
                start = time.time()
                
            ## changing values into tensor format

            mu_vals = data[:,:,mu_col].reshape(-1)
            mu_vals = mu_vals.type(torch.LongTensor)

            freq_bins = data[:,:,bin_col].reshape(-1)

            if n_covs == 1:
                covariate_vals = torch.squeeze(data[:,:,cov_col:]).unsqueeze(-1)
            else:
                covariate_vals = torch.squeeze(data[:,:,cov_col:])
                
            covariate_vals = covariate_vals.type(torch.FloatTensor)
    
            loss = svi.step(mu_vals, covariate_vals, freq_bins)

            losses.append(loss/data.shape[1])
            
            if batch_idx % divide_by == 0:
                if print_loss:
                    print("\t\t loss: " + str(loss/data.shape[1]), flush=True)

    model_filename = ".".join(output_filename.split(".")[:-1]) + ".model"
    param_filename = ".".join(output_filename.split(".")[:-1]) + ".params"
    
    output_dict = {}
    output_dict['KL']=KL
    output_dict['model']=model
    output_dict['guide']=guide
    
    with open(model_filename, 'wb') as handle:
        pickle.dump(output_dict, handle)
        
    pyro.get_param_store().save(param_filename)

    print("Finished training!", flush=True)

    ##############################post inference##############################
    
    result = post_analysis(neutral_sfs, mu_ref, n_bins, n_covs, losses, cov_sigma_prior = cov_sigma_prior)
    
    print("dumping file to output")
    with open(output_filename, 'wb') as f:
        pickle.dump(result, f)


def run_raklette_cov(loader, n_covs, num_epochs, neutral_sfs_filename, output_filename, lr, gamma, cov_sigma_prior = torch.tensor(0.1, dtype=torch.float32), mu_col = 0, bin_col = 1, cov_col = 2):
    #lr is initial learning rate

    print("running raklette", flush=True)

    # read neutral sfs
    sfs = pd.read_csv(neutral_sfs_filename, sep = "\t")
    bin_columns = []
    for i in range(len(sfs.columns) - 2):
        bin_columns.append(str(float(i)))
    neutral_sfs = torch.tensor(sfs[bin_columns].values)
    mu_ref = torch.tensor(sfs["mu"].values)
    n_bins = len(neutral_sfs[1]) - 1
    print("number of bins: " + str(n_bins), flush = True)

    KL = raklette.raklette_cov(neutral_sfs, n_bins, mu_ref, n_covs, cov_sigma_prior = cov_sigma_prior)
                
    model = KL.model
    guide = pyro.infer.autoguide.AutoNormal(model)

    #run inference
    pyro.clear_param_store()
    
    num_steps = num_epochs * len(loader)
    lrd = gamma ** (1 / num_steps)
    
    # run SVI
    optimizer = pyro.optim.ClippedAdam({"lr":lr, 'lrd': lrd})
#     optimizer = pyro.optim.Adam({"lr":lr})
    elbo = pyro.infer.Trace_ELBO(num_particles=1, vectorize_particles=True)
    svi = pyro.infer.SVI(model, guide, optimizer, elbo)
    losses = []
    
    start = time.time()
    divide_by = 1

    for epoch in range(num_epochs):
        print("epoch: " + str(epoch), flush = True)

        # Take a gradient step for each mini-batch in the dataset
        for batch_idx, data in enumerate(loader):
            if batch_idx % divide_by == 0:
                print("\t batch: " + str(batch_idx), flush = True)
                end = time.time()
                print("\t\t time from last batch: " + str(end - start), flush = True)
                start = time.time()
                
            ## changing values into tensor format

            mu_vals = data[:,:,mu_col].reshape(-1)
            mu_vals = mu_vals.type(torch.LongTensor)

            freq_bins = data[:,:,bin_col].reshape(-1)

            if n_covs == 1:
                covariate_vals = torch.squeeze(data[:,:,cov_col:]).unsqueeze(-1)
            else:
                covariate_vals = torch.squeeze(data[:,:,cov_col:])
                
            covariate_vals = covariate_vals.type(torch.FloatTensor)

            if batch_idx % divide_by == 0:
                end = time.time()
                print("\t\t time to get tensors: " + str(end - start), flush = True)
                start = time.time()
    
            loss = svi.step(mu_vals, covariate_vals, freq_bins)
        
            if batch_idx % divide_by == 0:
                end = time.time()
                print("\t\t time to get run svi step: " + str(end - start), flush = True)
                start = time.time()

            losses.append(loss/data.shape[1])
            
            if batch_idx % divide_by == 0:
                end = time.time()
                print("\t\t time to append loss: " + str(end - start), flush = True)
                start = time.time()

            if batch_idx % divide_by == 0:
                print("\t\t loss: " + str(loss/data.shape[1]), flush=True)

    model_filename = ".".join(output_filename.split(".")[:-1]) + ".model"
    param_filename = ".".join(output_filename.split(".")[:-1]) + ".params"
    
    output_dict = {}
    output_dict['KL']=KL
    output_dict['model']=model
    output_dict['guide']=guide
    
    with open(model_filename, 'wb') as handle:
        pickle.dump(output_dict, handle)
        
    pyro.get_param_store().save(param_filename)

    print("Finished training!", flush=True)

    ##############################post inference##############################
    
    result = raklette.post_analysis_cov(neutral_sfs, mu_ref, n_bins, n_covs, losses, cov_sigma_prior = cov_sigma_prior)
    
    print("dumping file to output")
    with open(output_filename, 'wb') as f:
        pickle.dump(result, f)

