import os
import glob
import sys
import numpy as np
import math
import random
import pickle
import csv


import pandas as pd
import dask
dask.config.set({'distributed.scheduler.allowed-failures': 0})
import dask.dataframe as dd
from dask.distributed import Client

import seaborn as sns
import matplotlib.pyplot as plt


sys.path.insert(0,'/home/djl34/lab_pd/bin')
import genomic

sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL/scripts')
sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL/simulation')

import post_analysis
import ml_raklette as mlr
import simulation_tools as simt

vep = "/home/djl34/bin/ensembl-vep/vep"
pd_data_dir = "/home/djl34/lab_pd/data"
KL_data_dir = "/home/djl34/lab_pd/kl/data"
scratch_dir = "/n/scratch/users/d/djl34/"

base_set = ["A", "C", "T", "G"]
all_chrom_set = [str(x) for x in range(1, 23)]
chrom_set = all_chrom_set

def get_mem_mb_small(wildcards, attempt):
    return attempt * 10000

per_generation_factor = 1.015 * 10 **-7

freq_breaks_2 = [-1, 1e-8, 0.5]
freq_breaks_9 = [-1, 1e-8, 1e-06, 2e-06, 4e-06, 2e-05, 5e-05, 5e-04, 5e-03, 0.5]
freq_breaks_10 = [-1, 1e-8, 1e-06, 2e-06, 4e-06, 2e-05, 5e-05, 5e-04, 5e-03, 5e-02, 0.5] 

