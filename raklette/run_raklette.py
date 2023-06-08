import sys
import pandas as pd
import torch

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL')

import raklette_updated

import pickle

KL_data_dir = "/home/djl34/lab_pd/kl/data"

########################################################################################################
# Create Dataset
########################################################################################################
class TSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples, header_all, features):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples // self.chunksize + 1
        self.header = header_all
        self.features = features

    def __getitem__(self, index):
        x = next(
            pd.read_csv(
                self.path,
                sep = "\t",
                skiprows=index * self.chunksize + 1,  #+1, since we skip the header
                chunksize=self.chunksize,
                names=self.header))

        x = x[self.features]
        x = torch.from_numpy(x.values)
        return x

    def __len__(self):
        return self.len


########################################################################################################
# Main Function
########################################################################################################
def run_raklette(loader, n_covs, n_genes, num_epochs, neutral_sfs_filename, output_filename, lr, gamma, fit_prior = True):
    #lr is initial learning rate

    print("running raklette", flush=True)

    # read neutral sfs
    sfs = pd.read_csv(neutral_sfs_filename, sep = "\t")
    bin_columns = []
    for i in range(5):
        bin_columns.append(str(float(i)))
    neutral_sfs = torch.tensor(sfs[bin_columns].values)
    mu_ref = torch.tensor(sfs["mu"].values)
    n_bins = len(neutral_sfs[1]) - 1
    print("number of bins: " + str(n_bins), flush = True)

    #define model and guide
    KL = raklette_updated.raklette(neutral_sfs, n_bins, mu_ref, n_covs, n_genes, fit_prior = fit_prior)
    
    if fit_prior == False:
        print("fitting with predefined prior for genes", flush = True)
        
        
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

    for epoch in range(num_epochs):
        print("epoch: " + str(epoch), flush = True)

        # Take a gradient step for each mini-batch in the dataset
        for batch_idx, data in enumerate(loader):
            gene_ids = data[:,:,2].reshape(-1)
            gene_ids = gene_ids.type(torch.LongTensor)

            mu_vals = data[:,:,0].reshape(-1)
            mu_vals = mu_vals.type(torch.LongTensor)

            freq_bins = data[:,:,1].reshape(-1)

            if n_covs == 0:
                loss = svi.step(mu_vals, gene_ids, None, freq_bins)
            else:
                # covariate_vals = data[:,:,3:].reshape((data.shape[0]*data.shape[1],) + data.shape[3:])
                covariate_vals = data[:,:,3:].reshape(-1).unsqueeze(0)
                covariate_vals = torch.transpose(covariate_vals, 0, 1)
                covariate_vals = covariate_vals.type(torch.LongTensor)

                loss = svi.step(mu_vals, gene_ids, covariate_vals, freq_bins)

            losses.append(loss/data.shape[1])

            if batch_idx % 10 == 0:
                print(batch_idx, flush=True)
                print(loss, flush=True)

    model_filename = ".".join(output_filename.split(".")[:-1]) + ".save"

    #save model parameters
    pyro.get_param_store().save(model_filename)

    #     # Tell the scheduler we've done one epoch.
    #     scheduler.step()

    #     print("[Epoch %02d]  Loss: %.5f" % (epoch, np.mean(losses)))

    print("Finished training!", flush=True)

    ##############################post inference##############################
    
    result = raklette_updated.post_analysis(neutral_sfs, mu_ref, n_bins, guide, n_covs, losses, fit_prior = fit_prior)

    with open(output_filename, 'wb') as f:
        pickle.dump(result, f)
