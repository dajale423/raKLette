import math
import torch
import fastDTWF
import numpy as np

# convert list of value to a list of indices where the value changes 
# and what those values are
def get_switch_points(x):
    switch_points = []
    switch_values = []
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            switch_points.append(i)
            switch_values.append(x[i])
    switch_points = np.array(switch_points, dtype=int)
    switch_points = -switch_points + switch_points[-1] + 1
    switch_points = np.concatenate([switch_points, [0]])
    switch_values = np.array(switch_values, dtype=int)
    switch_values = np.concatenate([[x[0]], switch_values])
    return switch_points, switch_values

# Tennessen et al. 2012 model of European population size
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3312577/
def tennessen_model(kk=1):
    N0 = math.ceil(7310*kk)
    N_old_growth = math.ceil(14474*kk)
    N_ooa_bn = math.ceil(1861*kk)
    N_ooa_bn_2 = math.ceil(1032*kk)
    N_growth_1 = math.ceil(9300*kk)
    N_growth_2 = math.ceil(512000*kk)

    t_old_growth = math.ceil(3880*kk)
    t_ooa_bn = math.ceil(1120*kk)
    t_growth_1 = math.ceil(715*kk)
    t_growth_2 = math.ceil(205*kk)

    r_growth_1 = (N_growth_1/N_ooa_bn_2)**(1/(t_growth_1-1))
    r_growth_2 = (N_growth_2/N_growth_1)**(1/(t_growth_2))

    N_set = np.array([N0] + [N_old_growth]*t_old_growth)
    N_set = np.append(N_set, [N_ooa_bn]*t_ooa_bn)
    N_set = np.append(N_set, N_ooa_bn_2*r_growth_1**np.arange(t_growth_1))
    N_set = np.append(N_set, N_growth_1*r_growth_2**np.arange(1, t_growth_2+1))
    return N_set.astype(np.int)

def tenn_sfs(shet, mu):
    """
    Generate the SFS for the Tennessen et al. 2012 model of European population size

    Parameters
    ----------
    shet : float
        The selection coefficient
    mu   : float
        The mutation rate
    
    Returns
    -------
    tenn_sfs : array_like
        The SFS for the Tennessen et al. 2012 model of European population size
    """
    tenn_N = tennessen_model()
    switch_points_tenn, pop_size_list_tenn = get_switch_points(tenn_N)
    shet = torch.tensor(shet, dtype=torch.float64)
    mu = torch.tensor(mu, dtype=torch.float64)
    tenn_sfs = fastDTWF.get_likelihood(
        pop_size_list=pop_size_list_tenn*2,    # population size at present
        switch_points=switch_points_tenn,      # sample individuals at present 
        sample_size=pop_size_list_tenn[-1]*2,  # sample the whole population
        s_het=shet,                            # selection coefficient
        mu_0_to_1=mu,                          # rate of mutating from "0" to "1"
        mu_1_to_0=mu,                          # rate of mutating from "1" to "0"
        dtwf_tv_sd=0.1,                        # controls accuracy of approximation
        dtwf_row_eps=1e-8,                     # controls accuracy of approximation
        sampling_tv_sd=0.05,                   # controls accuracy of approximation
        sampling_row_eps=1e-8,                 # controls accuracy of approximation
        no_fix=True,                           # condition on non-fixation
        sfs=False,
        injection_rate=0.0,
        use_condensed=True
    )
    return tenn_sfs