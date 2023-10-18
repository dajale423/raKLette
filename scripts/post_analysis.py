import pickle
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule
from pyro.infer import Predictive

def read_model(file_header):
    
    with open(file_header + ".model", 'rb') as handle:
        dictionary = pickle.load(handle)
    
    return dictionary

def get_n_cov(dictionary):
    
    guide = dictionary["guide"]
    KL = dictionary["KL"]
    model = dictionary["model"]

    beta_neut = KL.beta_neut
    beta_neut_full = KL.beta_neut_full
    beta_cov = guide.median()['beta_cov']
    
    return beta_cov.shape[0]

def get_beta_cov_trans(dictionary):
    
    guide = dictionary["guide"]
    KL = dictionary["KL"]
    model = dictionary["model"]

    beta_neut = KL.beta_neut
    beta_neut_full = KL.beta_neut_full
    beta_cov = guide.median()['beta_cov']
    
    beta_cov_trans = torch.cumsum(beta_cov, dim=-1)
    
    return beta_cov_trans

def get_n_sites(file_header):
    
    variants = "/n/scratch3/users/d/djl34/kl_input/"+ file_directory + file_header + "_length.tsv"
    
#     print(variants)
    
    df = pd.read_csv(variants, sep = "\t", header = None)
        
    return int(df.iloc[0, 0])

def freq_bin_to_AF_range(bin_num):
    if bin_num == 0:
        return "0"
    elif bin_num == 1:
        return "<1e-05 (singleton)"
    elif bin_num == 2:
        return "<1.7e-05 (doubleton)"
    elif bin_num == 3:
        return "<2.3e-05 (tripleton)"
    elif bin_num == 4:
        return "<3.6e-05"
    elif bin_num == 5:
        return "<8e-05"
    elif bin_num == 6:
        return "<5e-04"
    elif bin_num == 7:
        return "<5e-03"
    elif bin_num == 8:
        return "<5e-02"
    elif bin_num == 9:
        return "<0.5"
    
    