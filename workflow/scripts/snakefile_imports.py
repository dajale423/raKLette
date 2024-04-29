import os
import glob
import sys
import numpy as np
import math
import random
import pickle
import pandas as pd
import dask
dask.config.set({'distributed.scheduler.allowed-failures': 0})
import dask.dataframe as dd
from dask.distributed import Client

sys.path.insert(0,'/home/djl34/lab_pd/bin')
import genomic

vep = "/home/djl34/bin/ensembl-vep/vep"
pd_data_dir = "/home/djl34/lab_pd/data"
KL_data_dir = "/home/djl34/lab_pd/kl/data"
scratch_dir = "/n/scratch/users/d/djl34/"

base_set = ["A", "C", "T", "G"]
all_chrom_set = [str(x) for x in range(1, 23)]
chrom_set = all_chrom_set

def get_mem_mb_small(wildcards, attempt):
    return attempt * 10000