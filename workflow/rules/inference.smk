import sys
sys.path.insert(0,'/home/djl34/kl_git/scripts')
from snakefile_imports import *

##############################################################################################################################   
rule zscore_KL:
    input:
        rate = os.path.join(KL_data_dir, "{directory}/{chrom}.tsv"),
        neutral = os.path.join(KL_data_dir, "whole_genome/freq_bins/freq_bin_{bin_number}_all.tsv"),
    output:
        os.path.join(KL_data_dir, "{directory}/KL_freq_bin_{bin_number}_pseudocount_{pseudocount}_chrom_{chrom}.tsv"),
    resources:
        partition="short",
        runtime="0-0:20",
        cpus_per_task=4,
        mem_mb=7000
    run:
        with Client() as client:
            
            filename = input.rate
            ddf = dd.read_csv(filename, sep = "\t", dtype={'Spliceai_info': 'object'})
            ddf = ddf[~ddf["AF"].isna()]

            if int(wildcards.bin_number) == 9:
                freq_breaks = freq_breaks_9
            elif int(wildcards.bin_number) == 2:
                freq_breaks = freq_breaks_2
            
            ddf["Freq_bin"] = ddf['AF'].map_partitions(pd.cut, freq_breaks, labels = False)

            ddf["Freq_bin"] = ddf["Freq_bin"].astype(int)
            ddf["mu_index"] = ddf["mu_index"].astype(int)
            
            #from neutral SFS calculate probably of being polymorphic
            neutral = pd.read_csv(input.neutral, sep = "\t")
            neutral_sfs = neutral.drop(columns = ["mu", "sum"]).to_numpy()
            # neutral_sfs_beta = simt.multinomial_trans(neutral_sfs)

            def maximum_likelihood_KL(df, neutral_sfs = neutral_sfs, pseoudocount_num = 1):
                """
                Calculates the Maximum Likelihood KL from a dataframe of sites (with mu and frequency bin information)
                
                Arguments:
                ----------
                    df (pandas dataframe): a dataframe to calculate the KL over
                    
                Returns:
                --------
                    KL values (numpy float): the calculated KL
                """
                array = np.zeros(neutral_sfs.shape)
                neutral_sfs_beta = simt.multinomial_trans(neutral_sfs)
            
                array_positions = pd.DataFrame(df.groupby(["mu_index", "Freq_bin"]).size()).reset_index()
            
                for index, row in array_positions.iterrows():
                    array[int(row['mu_index']), row['Freq_bin']] += row[0]
            
                pseudocount_array = array + pseoudocount_num * neutral_sfs
                winsfs_test = mlr.WinSFS(data = pseudocount_array, mut_offset = neutral_sfs_beta)
            
                winsfs_test.ml_optim(jac=True, beta_max=100, verbose=False)
            
                return winsfs_test.KL(20) ##20 means 20 mu bin, or mu = 0.236

            df_kl = ddf.groupby(['region']).apply(lambda x: maximum_likelihood_KL(x, neutral_sfs = neutral_sfs, pseoudocount_num = int(wildcards.pseudocount))).compute()

            df_groupby = pd.DataFrame(df_kl).reset_index()
            df_groupby = df_groupby.rename({0: "max_likelihood_KL"}, axis = 1)
            
            df_groupby.to_csv(output[0], sep = "\t")

rule calculate_loglikelihood_cdf:
    input:
        rate = os.path.join(KL_data_dir, "{directory}/{chrom}.tsv"),
        neutral = os.path.join(KL_data_dir, "whole_genome/freq_bins/freq_bin_{bin_number}_all.tsv"),
    output:
        os.path.join(KL_data_dir, "{directory}/LLR_freq_bin_{bin_number}_reverse_{reverse}_chrom_{chrom}.tsv"),
    resources:
        partition="short",
        runtime="0-1:00",
        cpus_per_task=4,
        mem_mb=7000
    run:
        with Client() as client:
            
            filename = input.rate
            ddf = dd.read_csv(filename, sep = "\t", dtype={'Spliceai_info': 'object'})
            ddf = ddf[~ddf["AF"].isna()]

            if int(wildcards.bin_number) == 9:
                freq_breaks = freq_breaks_9
            elif int(wildcards.bin_number) == 2:
                freq_breaks = freq_breaks_2
            
            ddf["Freq_bin"] = ddf['AF'].map_partitions(pd.cut, freq_breaks, labels = False)

            ddf["Freq_bin"] = ddf["Freq_bin"].astype(int)
            ddf["mu_index"] = ddf["mu_index"].astype(int)
            
            #from neutral SFS calculate probably of being polymorphic
            neutral = pd.read_csv(input.neutral, sep = "\t")
            neutral_sfs = neutral.drop(columns = ["mu", "sum"]).to_numpy()
            # neutral_sfs_beta = simt.multinomial_trans(neutral_sfs)

            def log_likelihood_ratio(df, neutral_sfs = neutral_sfs, reverse = False):
                """
                Calculates the Maximum Likelihood KL from a dataframe of sites (with mu and frequency bin information)
                
                Arguments:
                ----------
                    df (pandas dataframe): a dataframe to calculate the KL over
                    
                Returns:
                --------
                    KL values (numpy float): the calculated KL
                """

                ## array is observed polymorphism SFS
                array = np.zeros(neutral_sfs.shape)
                array_positions = pd.DataFrame(df.groupby(["mu_index", "Freq_bin"]).size()).reset_index()
            
                for index, row in array_positions.iterrows():
                    array[int(row['mu_index']), row['Freq_bin']] += row[0]
            
                winsfs_test = mlr.WinSFS(data = array, neutral_sfs = neutral_sfs)
            
                return winsfs_test.log_likelihood_ratio_cdf(reverse = reverse)
            
            df_llr = ddf.groupby(['region']).apply(lambda x: log_likelihood_ratio(x, neutral_sfs = neutral_sfs, 
                                                                                  reverse = bool(int(wildcards.reverse)))).compute()
            df_groupby = pd.DataFrame(df_llr).reset_index()
            df_groupby = df_groupby.rename({0: "LLR"}, axis = 1)            
            df_groupby.to_csv(output[0], sep = "\t")


rule calculate_LOEUF:
    input:
        rate = os.path.join(KL_data_dir, "{directory}/{chrom}.tsv"),
        neutral = os.path.join(KL_data_dir, "whole_genome/freq_bins/freq_bin_9_all.tsv"),
    output:
        os.path.join(KL_data_dir, "{directory}/LOEUF_chrom_{chrom}.tsv"),
    resources:
        partition="short",
        runtime="0-0:20",
        cpus_per_task=1,
        mem_mb=2000
    run:
            filename = input.rate
            ddf = dd.read_csv(filename, sep = "\t", dtype={'Spliceai_info': 'object'})
            ddf = ddf[~ddf["AF"].isna()]
                        
            neutral_sfs = pd.read_csv(input.neutral, sep = "\t")
            
            neutral_sfs = neutral_sfs.rename({"0.0": "p_monomorphic"}, axis = 1)
            neutral_sfs["p_polymorphic"] = 1 - neutral_sfs["p_monomorphic"]

            rate = ddf.merge(neutral_sfs[["mu", "p_polymorphic"]], how = "left", on = "mu")
            rate["obs_polymorphic"] = rate["Freq_bin_9"].astype(bool).astype(int)
            rate_pergene = rate[["region", "p_polymorphic", "obs_polymorphic"]].groupby("region").sum().compute()
            
            rate_pergene = pd.DataFrame(rate_pergene).reset_index()
            
            from scipy.stats import poisson

            def get_upper_bound(count, expected):

                k = count

                lambd_list = np.linspace(0,2,2001)
                pmf = [poisson.pmf(k, lambd * expected) for lambd in lambd_list]
                dist_df = pd.DataFrame(zip(lambd_list, pmf), columns = ["Lambda", "pmf"])

                dist_df["pmf"] = dist_df["pmf"]/dist_df["pmf"].sum()
                dist_df["cdf"] = np.cumsum(dist_df["pmf"])

                dist_df_upper = dist_df[(dist_df["cdf"] < 0.95)]

                upper_bound = dist_df_upper[dist_df_upper["cdf"] == dist_df_upper["cdf"].max()]

                return upper_bound
            
            rate_pergene['LOEUF'] = rate_pergene.apply(lambda row: get_upper_bound(row["obs_polymorphic"], row["p_polymorphic"]), axis = 1)
            
#             rate_pergene["LOEUF"] = rate_pergene["LOEUF"].str.split(expand = True)[4]  
#             rate_pergene["LOEUF"] = rate_pergene["LOEUF"].astype(float)

            rate_pergene.to_csv(output[0], sep = "\t", index = None) 