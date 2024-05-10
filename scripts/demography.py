import pandas as pd
import glob
import numpy as np
from scipy.stats import binom

def read_sfs(sfs_filename): 
    sfs = pd.read_csv(sfs_filename, sep = "\t")   
    return sfs

def read_sfs_sum(directory, file_header, seed_list, tail = ""):
    first = True

    for i in seed_list:
        filename = file_header + str(i) + tail + ".tsv"

        sfs = read_sfs(directory + filename)

        if first == True:
            sfs_combined = sfs
            first = False
        else:
            sfs_combined = pd.concat([sfs_combined, sfs])

    sfs = pd.DataFrame(sfs_combined.groupby("Counts").sum()).reset_index()
    
    return sfs
    
def get_sing_over_seg(sfs_filename):
    sfs = read_sfs(sfs_filename)
    return sfs[sfs["Counts"] == 1]["Number"].sum()/sfs[sfs["Counts"] > 0]["Number"].sum()

def get_monomorphic(sfs_filename):
    sfs = read_sfs(sfs_filename)
    
    monomorphic = sfs.loc[sfs["Counts"] == 0].iloc[0]["Number"]/sum(sfs["Number"])
    
    return monomorphic

def get_ac(sfs_filename, ac,  an):
    sfs = read_sfs(sfs_filename)
    
    proportion = sfs.loc[sfs["Counts"] == ac].iloc[0]["Number"]/sum(sfs["Number"])
    
    mac = an - ac
    if len(sfs.loc[sfs["Counts"] == mac]) > 0:
        proportion += sfs.loc[sfs["Counts"] == mac].iloc[0]["Number"]/sum(sfs["Number"])
    
    return proportion

def fold_sfs(sfs, max_AC):
    
    sfs = sfs.copy()
    
    sfs["MAC"] = [max_AC - x for x in sfs["Counts"]]
    
    sfs["MAC"] = sfs[['Counts','MAC']].min(axis=1)
    
    sfs = sfs.groupby("MAC").sum()
    
    sfs = pd.DataFrame(sfs)
    
    sfs = sfs.reset_index()
    
    sfs.drop("Counts", axis = 1, inplace = True)
    
    return sfs

def binarize(sfs, column_name):
    
    sfs = sfs.copy()
    
    sfs["Seg"] = np.where(sfs[column_name] == 0, 0, 1)
    
    sfs = sfs.groupby("Seg").sum().reset_index()
    
    sfs.drop(column_name, axis = 1, inplace = True)
    
    return sfs

# defining function to check price
def get_bin(x, base):
    if x == 0:
        return -1
    else:
        return int(np.log(x)/np.log(base))

def make_log_bin(sfs, base, column_name):
    sfs = sfs.copy()
    
    sfs["bin"] = [get_bin(x, base) for x in sfs[column_name]]
    sfs = pd.DataFrame(sfs.groupby("bin").sum()).reset_index()
    
    sfs.drop(column_name, axis = 1, inplace = True)
    
    return sfs

def make_sfs_count(sfs_filename, max_AC):
    
    sim_df = read_sfs(sfs_filename)
    sim_df = create_mac(sim_df, max_AC)
    
    list_counts = []
    
    total_count = sim_df["Number"].sum()
    
    # add monomorphic prop
    monomorphic = sim_df[sim_df["MAC"] == 0]["Number"].iloc[0]
    
    list_counts.append(monomorphic)
    
    polymorphic = total_count - monomorphic
#     list_counts.append(polymorphic)
    
    sum_rare_counts = 0
    for i in range(1, 5):
        ac = sim_df[sim_df["MAC"] == i]["Number"].iloc[0]        
        list_counts.append(ac)
        sum_rare_counts += ac
        
    common_counts = polymorphic - sum_rare_counts
    list_counts.append(common_counts)
        
    return list_counts

def make_simulations_df(dir_name, growth_col, seed_col, mu_col, scaling_col):
    file_list = glob.glob("/net/home/dlee/brca1/data/dan_sim/SFS_output_v2.6.1_recurrent_slow/"+ dir_name +"/sampleSFS_size_113770/*.tsv")

    sim_df = pd.DataFrame(file_list, columns = ["dir_filename"])

    sim_df["filename"] = sim_df["dir_filename"].str.split("/", expand = True)[10]

    sim_df["growth"] = sim_df["filename"].str.split("_", expand = True)[growth_col]
    sim_df["seed"] = sim_df["filename"].str.split("_", expand = True)[seed_col]
    sim_df["seed"] = sim_df["seed"].str.split(".", expand = True)[0]
    sim_df["mu"] = sim_df["filename"].str.split("_", expand = True)[mu_col]
    sim_df["scaling_factor"] = sim_df["filename"].str.split("_", expand = True)[scaling_col]
    
    return sim_df

def add_to_data_df_after_filtering(data_df, sim_df, mu_rate, scaling, string, growth_rate = False):    
    #filter by mu
    sim_df = sim_df[sim_df["mu"] == str(mu_rate)]
    
    if growth_rate:    
        sim_df_fit = sim_df[sim_df["growth"] == str(growth_rate)]
    else:
        sim_df_fit = sim_df.copy()
    
    sim_df_fit["scaling_factor"] = sim_df_fit["scaling_factor"].astype(float)
    sim_df_fit = sim_df_fit[sim_df_fit["scaling_factor"] == float(scaling)]
    
#     print(sim_df_fit)

    sim_df_fit["sfs_count"] = [make_sfs_count(x, 113770) for x in sim_df_fit["dir_filename"]]

    data_df["sim_count_" + string] = [sum(x) for x in zip(*list(sim_df_fit["sfs_count"]))]
    
    data_df["sim_prop_" + string] = [x/data_df["sim_count_" + string].sum() for x in data_df["sim_count_" + string]]

    seg_total = data_df["sim_count_" + string].sum() - data_df[data_df["MAC_nfe"] == "0.0"]["sim_count_" + string].iloc[0]

    data_df["sim_seg_prop_" + string] = [x/seg_total for x in data_df["sim_count_" + string]]
    
    return data_df

### downsample
def downsample(pop_size, sample_size, sfs):
    
    sfs = sfs.copy()
    
    expected_freq = sfs["Counts"]/pop_size

    sfs["Sample_Counts"] = np.array(np.random.binomial(sample_size, expected_freq))

    sfs_sample = sfs.groupby("Sample_Counts").sum()
    sfs_sample.reset_index(inplace = True)
    sfs_sample.drop("Counts", axis = 1, inplace = True)
    
    sfs_sample = sfs_sample.rename({"Sample_Counts": "Counts"}, axis = 1)
    
    return sfs_sample

def downsample_expected(pop_size, sample_size, sfs):
    counts = []
    for i in range(sample_size + 1):
        counts.append(sum(binom.pmf(i, sample_size, sfs["Counts"]/pop_size) * sfs["Number"]))
        
    return pd.DataFrame(zip(range(sample_size + 1), counts), columns = ["Counts", "Number"])

def simulate_multiple_downsample(sfs, pop_size, sample_size, iterations):
    first = True

    for i in range(iterations):
        sfs_sample = downsample(pop_size, sample_size, sfs)

        if first == True:
            sfs_sample_combined = sfs_sample
            first = False
        else:
            sfs_sample_combined = pd.concat([sfs_sample_combined, sfs_sample])

    sfs_sample_sum = pd.DataFrame(sfs_sample_combined.groupby("Counts").sum()).reset_index()
    
    sfs_sample_sum["Number"] = sfs_sample_sum["Number"]/iterations 
    
    return sfs_sample_sum

def max_likelihood_SFS(counts, expected_eta):
    if pd.isna(counts):
        return 0
    else:
        return counts * np.log(expected_eta)