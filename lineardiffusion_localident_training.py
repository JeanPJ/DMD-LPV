#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:21:46 2024

@author: jeanpj
"""

import numpy as np
import scipy.linalg as sla
import scipy.io as io
from lpvs_ident import *

load_dict = io.loadmat(open("lineardiffusion_data_local.mat",'rb'))


u_signal_list = load_dict['u_signal_list']
T_plot_list = load_dict['T_plot_list']

p_list = load_dict['p_list']



LTI_list = []

from lineardata_plot import *
from lpvs_ident import *

def k_fun(p):
    
    
    p = p.item()
    
    #return np.atleast_2d([1,p,p**2]).T
    #return np.atleast_2d([1,p,p**2,p**3]).T
    return np.atleast_2d([1,p,p**2,p**3,p**4]).T
    #return np.atleast_2d([np.exp(p),np.sin(p)]).T


n_in = 1
n_p = 1



diffusion_black_box_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)

T0 = 0*np.ones([number_of_states,1])
T_plot_init_list = []
for i,p in enumerate(p_list):
    T_plot_init = np.hstack([T0,T_plot_list[i][:,:-1]])
    
    T_plot_init_list += [T_plot_init]
    
    
    
print(u_signal_list[0].shape)
print(T_plot_list[0].shape)

from lineardata_plot import *


rank_list = [1,5,10,15,20,25,30,35,40,45,0]


for rank in rank_list:
    
    diffusion_black_box_model.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = rank)




    T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])
    e = diffusion_black_box_model.get_error_from_series(T_plot_init.T, u_signal.T, p_signal.T, T_plot.T)

    print(e,"error for rank (proc and pod)",rank)