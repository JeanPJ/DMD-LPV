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

load_dict = io.loadmat(open("nonpolydiffusion_data_local99states.mat",'rb'))


u_signal_list = load_dict['u_signal_list']
T_plot_list = load_dict['T_plot_list']

p1_list = load_dict['p1_list']
p2_list = load_dict['p2_list']



LTI_list = []

from nonpolydata_plot import *
from lpvs_ident import *

w_elm = io.loadmat("elm_weights.mat")['W']


def generate_p_list(p1_list,p2_list):
    
    p_list = []
    
    for i,p1 in enumerate(p1_list.flatten()):
        for j,p2 in enumerate(p2_list.flatten()):
            p_list += [np.array([p1,p2])]
    
    
    
    
    return np.array(p_list).T




def elm_basis(p,Nelm,W):
    
    
    elm_vec = np.empty([Nelm,p.shape[1]])
    
    
    bias = np.ones([1,p.shape[1]])
    
    p_and_bias = np.vstack([bias,p])
    
    
    for i in range(Nelm):
        
        elm_vec[i] = np.tanh(W[i]@p_and_bias)
    return elm_vec


n_in = 1
n_p = 2

Nelm = 15

k_fun = lambda p:elm_basis(p,Nelm,w_elm)


diffusion_black_box_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)

T0 = 0*np.ones([number_of_states,1])
T_plot_init_list = []

p_list = generate_p_list(p1_list, p2_list)
for i,p in enumerate(p_list):
    T_plot_init = np.hstack([T0,T_plot_list[i][:,:-1]])
    
    T_plot_init_list += [T_plot_init]
    
    
    
print(u_signal_list[0].shape)
print(T_plot_list[0].shape)



rank_list = [10]


for rank in rank_list:
    
    diffusion_black_box_model.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = rank)




    T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])
    e = diffusion_black_box_model.get_error_from_series(T_plot_init.T, u_signal.T, p_signal.T, T_plot.T)

    print(e,"error for rank (proc and pod)",rank)