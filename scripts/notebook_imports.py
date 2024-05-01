import os
import sys
import csv
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(0,'/home/djl34/lab_pd/bin')
import genomic

# sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL/scripts')
sys.path.insert(0, '/home/djl34/lab_pd/kl/git/KL/simulation')

import post_analysis
import ml_raklette as mlr
import simulation_tools as simt

pd.set_option('display.max_columns', None)

all_chrom_set = [str(x) for x in range(1, 23)]
chrom_set = all_chrom_set
per_generation_factor = 1.015 * 10 **-7
freq_breaks_9 = [-1, 1e-8, 1e-06, 2e-06, 4e-06, 2e-05, 5e-05, 5e-04, 5e-03, 0.5]

pd_data_dir = "/home/djl34/lab_pd/data"
aso_data_dir = "/home/djl34/lab_pd/aso/data"
KL_data_dir = "/home/djl34/lab_pd/kl/data"
scratch_dir = "/n/scratch/users/d/djl34"