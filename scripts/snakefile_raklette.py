import os
import sys
import glob
import numpy as np
import pandas as pd
import math
import sys
import random
import pickle

import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL/scripts')
import raklette
from run_raklette import run_raklette
from run_raklette import run_raklette_cov
from run_raklette import TSVDataset



def snakefile_raklette_cov(input_filename, neutral_sfs_filename, input_length_file, output_filename, chunksize, epoch, cov_prior, learning_rate, gamma):
    
    df = pd.read_csv(input_length_file, sep = "\t", header = None)
    nb_samples = df[0][0]
    nb_features = df[0][1] - 2

    print("number of samples: " + str(nb_samples), flush = True)

    if nb_samples == 0:
        f = open(output_filename, "w")
        f.write("no sample")
        f.close()
    else:        
        with open(input.variants) as f:
            first_line = f.readline()
        header = first_line.split("\t")

        print("number of chunks " + str(nb_samples/chunksize), flush = True)

        dataset = TSVDataset(input_directory, chunksize=chunksize, nb_samples = nb_samples, header_all = header, features = header)
        loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

        num_epochs = int(epoch)
        cov_prior = float(cov_prior)

        #lets run raklette
        run_raklette_cov(loader, nb_features, num_epochs, neutral_sfs_filename, output_filename, 
                     float(learning_rate), float(gamma), 
                         cov_sigma_prior = torch.tensor(cov_prior, dtype=torch.float32))
