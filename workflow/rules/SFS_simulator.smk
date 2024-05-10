import os
import sys
import csv
import glob
import gzip 
import numpy as np

import pandas as pd
import dask
import dask.dataframe as dd
import math
import sys

sys.path.insert(0, "/home/djl34/kl_git/scripts")

import demography as dm
from others import round_sig

################################################################################################################################

output_dir = "/home/djl34/kl_git/results"
code_dir = "/home/djl34/kl_git/scripts"
simulator = code_dir + "/simDoSe_v2.6.3_Uindex.py"

# for standard growth rate fit to gnomADv2.1.1 data
standard_growth_rate = True

# demography options are gao, equilibrium
demography = "gao"

# some_list = range(1, 41, 2)
digit_list = range(1, 10, 1)

# sample_size_list = np.geomspace(10000, 1000000, num=9)
# sample_size_list = [int(x) for x in sample_size_list]
sample_size_list = [500000 * 2]
# sample_size_list = [10000]

selection_list = [0.0, 0.1, 0.01, 0.001]

# for i in range(1,4):
#     selection_list.extend([x*(10**(-1*i)) for x in mut_list])
# selection_list.extend(np.geomspace(1.0, 0.001, num=16))

mu_list = [2e-09, 6e-09, 2e-08, 6e-08, 2e-07]

seed_list = range(10)
seed_num = 0

growth_list = [0.0057]
growth_beta_list = [1.122]

scaling_list = [1]

L = str(float(5.0))
Recurrent = 1

output_sample_list = True

#####################polish up the lists##########################################
scaling_list = [float(x) for x in scaling_list]
mu_list = [round_sig(x, sig = 2) for x in mu_list]
selection_list = [round_sig(x, sig = 2) for x in selection_list]

print(selection_list)

output_list = []

if Recurrent == 0:
    recurrent_directory = "SFS_output_v2.6.1"
elif Recurrent == 1:
    recurrent_directory = "SFS_output_v2.6.1_recurrent_slow"
    
if standard_growth_rate == True:
    if demography == "supertennessen":
        growth_list = [0.032]
    if demography == "gao":
        growth_list = [0.038]
    if demography == "gaussian":
        growth_list = [0.00016]

###################################################################################################################################################
  
if demography == "gao":
    file_header = "SFS_gao_2N_20000"
else:
    file_header = ""

filename = os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/raw_SFS/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_0.0057_growthbeta_1.122_scalingfactor_1.0_seed_{seed_num}.tsv")

input_list = [filename.format(mu = mu, selection = selection, seed_num = seed_num) for mu in mu_list for selection in selection_list for seed_num in seed_list]

filename = os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/sample_1000000/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_0.0057_growthbeta_1.122_scalingfactor_1.0.tsv")

input_list = [filename.format(mu = mu, selection = selection) for mu in mu_list for selection in selection_list]

# rule all:
#     input:
#         input_list
#         [os.path.join(output_dir, recurrent_directory + "/" + demography + add +"/raw_SFS/"+ file) for file in output_list_sample]
    

######################################################Get Sample SFS####################################################
def get_final_2N(demography, growth, mu, growth_beta, seed):
    
    directory = "/home/djl34/lab_pd/simulator/data/SFS_output_v2.6.1_recurrent_slow/"+ demography +"/raw_SFS/summary_stats/"
    
    if demography == "gao":
        file_header = "stats_gao_2N_20000_"
        
    file_header = file_header + "Slinear__-0.0_h_0.5_mu_" + str(mu) 
    file_header = file_header + "_L_5.0_growth_" + str(growth) + "_growthbeta_" + str(growth_beta)
    file = file_header + "_scalingfactor_1.0_seed_" + str(seed) + ".txt"

    with open(directory + file) as f:
        lines = f.readlines()
        
        for i in lines:
            if "Final size 2N" in i:
                return int(i.split()[-1])
            
import os.path
            
def get_sample_sfs(demography, selection, growth, mu, growth_beta, seed_list, sample_size, unfolded = False):


    directory = "/home/djl34/kl_git/results/SFS_output_v2.6.1_recurrent_slow/"+ demography +"/raw_SFS/"

    if demography == "gao":
        file_header = "SFS_gao_2N_20000_"
    
    file_header = file_header + "Slinear__-" + selection + "_h_0.5_mu_" + str(mu) 
    file_header = file_header + "_L_5.0_growth_" + str(growth) + "_growthbeta_" + str(growth_beta)
    file_header = file_header + "_scalingfactor_1.0_seed_"
    
    sample_size = int(sample_size)
    growth = float(growth)
    growth_beta = float(growth_beta)

    sfs = dm.read_sfs_sum(directory, file_header, seed_list)

    pop_size = get_final_2N(demography, growth, mu, growth_beta, 0)

    sfs_sample = dm.downsample_expected(pop_size, sample_size, sfs)
    
    if unfolded == False:
        sfs_sample_sum_folded = dm.fold_sfs(sfs_sample, sample_size)
        # sfs_sample_sum_folded.to_csv(output_filename, sep = "\t", index = None)
        return sfs_sample_sum_folded
    else:
        # sfs_sample.to_csv(output_filename, sep = "\t", index = None)
        return sfs_sample
###################################################### Run for Equilibrium ####################################################
wildcard_constraints:
    sample_size="\d+"


# sample_size_list = range(1, 101, 20)
# sample_size_list = [1000 * x for x in sample_size_list]
sample_size_list = [1]

rule run_equilibrium_syn:
    input:
        os.path.join(code_dir, "simDoSe_v2.6.3_Uindex.py")
    output:
        os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/equilibrium/raw_SFS/SFS_equilibrium_2N_{sample_size}_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_scalingfactor_1_seed_{seed_num}.tsv")
    shell:
        "python {simulator} -U {wildcards.mu} -L 100000 -S -{wildcards.selection} -H 0.5 -D equilibrium -N {wildcards.sample_size} --recurrent 1 --Rkernel slow --linearU --seed {wildcards.seed_num} --linearS 1 --samplesize 1000"


########################################################## Run for Gao ########################################################

#gao and keinan 2016 has final growth rate of 0.023
#2N_20000 indicates the initial population size

rule gao_syn_gnomADv2:
    input:
        os.path.join(code_dir, "simDoSe_v2.6.3_Uindex.py")
    output:
        os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/raw_SFS/SFS_gao_2N_20000_Slinear__-0.0_h_0.5_mu_{mu}_L_5.0_growth_{growth}_scalingfactor_{scaling}_seed_{seed_num}.tsv")
    wildcard_constraints:
        growth="[+-]?([0-9]*[.])?[0-9]+"
    shell:
        "python {simulator} -U {wildcards.mu} -L 100000 -S -0.0 -H 0.5 -D gao --recurrent 1 --Rkernel slow --linearU --seed {wildcards.seed_num} --linearS 1 --samplesize 113770 --scaling-factor {wildcards.scaling} -G {wildcards.growth}"
        
rule gao_syn_gnomADv2_growthbeta:
    input:
        # os.path.join(code_dir, "simDoSe_v2.6.3_Uindex.py")
    output:
        os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/raw_SFS/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_{growth}_growthbeta_{growth_beta}_scalingfactor_{scaling}_seed_{seed_num}.tsv")
    wildcard_constraints:
        growth="[+-]?([0-9]*[.])?[0-9]+",
        seed_num ="[+-]?([0-9]*[.])?[0-9]+"
    resources:
        partition="short",
        runtime="0-12:00",
        cpus_per_task=1,
        mem_mb=1000
    shell:
        "python {simulator} -U {wildcards.mu} -L 100000 -S -{wildcards.selection} -H 0.5 -D gao --recurrent 1 --Rkernel slow --linearU --seed {wildcards.seed_num} --linearS 1 --samplesize 113770 --scaling-factor {wildcards.scaling} -G {wildcards.growth} --growth_beta {wildcards.growth_beta}"
        
rule gao_syn_gnomADv2_growthbeta_get_sample:
    input:
        [os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/raw_SFS/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_{growth}_growthbeta_{growth_beta}_scalingfactor_{scaling}_seed_"+ str(i) +".tsv") for i in range(10)]
    output:
        os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/sample_{sample_size}/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_{growth}_growthbeta_{growth_beta}_scalingfactor_{scaling}.tsv")
    resources:
        partition="short",
        runtime="0-12:00",
        cpus_per_task=1,
        mem_mb=1000
    run:
        sfs_sample = get_sample_sfs("gao", wildcards.selection, wildcards.growth, wildcards.mu, wildcards.growth_beta, range(int(10)), wildcards.sample_size)
        sfs_sample.to_csv(output[0], sep = "\t", index = None)
        
# rule gao_syn_gnomADv2_growthbeta_get_sample_unfolded:
#     input:
#         [os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/raw_SFS/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_{growth}_growthbeta_{growth_beta}_scalingfactor_{scaling}_seed_"+ str(i) +".tsv") for i in range(seed_num)]
#     output:
#         os.path.join(output_dir, "SFS_output_v2.6.1_recurrent_slow/gao/raw_SFS/SFS_gao_2N_20000_Slinear__-{selection}_h_0.5_mu_{mu}_L_5.0_growth_{growth}_growthbeta_{growth_beta}_scalingfactor_{scaling}_seed_{seed_num}_sample_{sample_size}_unfolded.tsv")
#     run:
#         sfs_sample = get_sample_sfs("gao", wildcards.selection, wildcards.growth, wildcards.mu, wildcards.growth_beta, range(int(wildcards.seed_num)), wildcards.sample_size, unfolded = True)
        