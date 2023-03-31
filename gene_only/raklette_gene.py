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

import dask.dataframe as dd
from dask.distributed import Client

KL_data_dir = "/home/djl34/lab_pd/kl/data"

########################################################################################################
# Create Dataset
########################################################################################################
class TSVDataset(Dataset):
    def __init__(self, path, chunksize, nb_samples, header_all, features):
        self.path = path
        self.chunksize = chunksize
        self.len = nb_samples // self.chunksize
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
        x["e_module"] = x["e_module"] + 1
        x = torch.from_numpy(x.values)
        return x

    def __len__(self):
        return self.len
    
########################################################################################################
# Main Function
########################################################################################################
def run_raklette(loader, n_covs, n_genes, num_epochs, neutral_sfs_filename, output_filename):
    
    print("running raklette")
    
    # read neutral sfs
    sfs = pd.read_csv(neutral_sfs_filename, sep = "\t")
    bin_columns = []
    for i in range(5):
        bin_columns.append(str(i) + "_bin")
    neutral_sfs = torch.tensor(sfs[bin_columns].values)
    mu_ref = torch.tensor(sfs["mu"].values)
    
    n_bins = len(neutral_sfs[1]) - 1

    #define model and guide
    KL = raklette_updated.raklette(neutral_sfs, n_bins, mu_ref, n_covs, n_genes)
    model = KL.model
    guide = pyro.infer.autoguide.AutoNormal(model)

    #run inference
    pyro.clear_param_store()
    # run SVI
    adam = pyro.optim.Adam({"lr":0.005})
    elbo = pyro.infer.Trace_ELBO(num_particles=1, vectorize_particles=True)
    svi = pyro.infer.SVI(model, guide, adam, elbo)
    losses = []

#     num_epochs = 1

    for epoch in range(num_epochs):
        # Take a gradient step for each mini-batch in the dataset
        for batch_idx, data in enumerate(loader):
            gene_ids = data[:,:,2].reshape(-1)
            gene_ids = gene_ids.type(torch.LongTensor)

            mu_vals = data[:,:,0].reshape(-1)
            mu_vals = mu_vals.type(torch.LongTensor)

            loss = svi.step(mu_vals, gene_ids, None, data[:,:,1].reshape(-1))
    #         if y is not None:
    #             y = y.type_as(x)
    #         loss = svi.step(x, y)
            losses.append(loss)

    #     # Tell the scheduler we've done one epoch.
    #     scheduler.step()

    #     print("[Epoch %02d]  Loss: %.5f" % (epoch, np.mean(losses)))

    print("Finished training!")

    # for step in tqdm(range(n_steps)): # tqdm is just a progress bar thing 
    #     loss = svi.step(mu_vals, gene_ids, None, sample_sfs)
    #     print(loss)
    #     losses.append(loss)

    ##############################post inference##############################
    
    result = raklette_updated.post_analysis(neutral_sfs)

    result.to_csv(output_filename, sep = "\t")
