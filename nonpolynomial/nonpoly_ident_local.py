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
import pickle

load_dict = io.loadmat(open("nonpolydiffusion_data_local99states.mat",'rb'))
#load_dict = pickle.load(open("Improviso.pickle",'rb'))


u_signal_list = load_dict['u_signal_list']
T_plot_list = load_dict['T_plot_list']

p_list = np.array(load_dict['p_list'])




from nonpolydata_plot import *
from lpvs_ident import *

w_elm = io.loadmat("elm_weights.mat")['W']



 




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


for i,p in enumerate(p_list):
    T_plot_init = np.hstack([T0,T_plot_list[i][:,:-1]])
    
    T_plot_init_list += [T_plot_init]
    
    
    
print(u_signal_list[0].shape)
print(T_plot_list[0].shape)



rank_list = [10]

pr_rank = 10

p_list = p_list.T

reg = 5e-2

for rank in rank_list:
    
    diffusion_black_box_model.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = pr_rank,reg =reg)




    T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])
    e = diffusion_black_box_model.get_error_from_series(T_plot_init.T, u_signal.T, p_signal.T, T_plot.T)

    print(e,"error for rank (proc and pod)",rank)
    
    
    
    
    
    load_dict = io.loadmat(open("nonpolydiffusion_test.mat",'rb'))
    
    u_test = load_dict['u_signal'].T
    p_test = load_dict['p_signal']
    T_test = load_dict['T_plot']
    
    
    simtime = T_test.shape[0]
    
    
    T_sim = np.empty(simtime)
    T_ls = np.empty(simtime)
    T0 = np.zeros([number_of_states, 1])
    
    z0 = diffusion_black_box_model.get_state_reduction(T0)
    z = z0
    
    #z_ls = T0
    
    
    T_test_init = np.vstack([T0.T,T_test[:-1,:]])
    
    
    e_test = diffusion_black_box_model.get_error_from_series(T_test_init.T,u_test.T,p_test,T_test.T)
    #e_ls = golden_model.get_error_from_series(T_test_init.T,u_test.T,p_test,T_test.T)
    print("test error:",e_test)
    
    for k in range(simtime):
    
        z = diffusion_black_box_model.update_latent(z, u_test[k], p_test[:,k:k+1])
        #z_ls = golden_model.update_latent(z_ls, u_test[k], p_test[:,k:k+1])
        T_sim[k] = (diffusion_black_box_model.T@z).flatten()[-1]
        #T_ls[k] = z_ls.flatten()[-1]
        
    # plt.plot(T_sim,label = f"ELM DMD-LPV POD rank = {pod} Pr = {red_order}")
    # plt.plot(T_ls,label = f"ELM LPV (LS)")
    # plt.plot(T_test[:, -1],label = "Test Signal")
    # plt.xlabel("timesteps")
    # plt.ylabel("T (global)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    
    fig, axs = plt.subplots(4)
    Ts = 1e-2
    fig.suptitle('ELM DMD-LPV Simulation over Test Data (local)')
    axs[0].plot(T_sim,label = f"ELM DMD-LPV")
    #axs[0].plot(T_ls,label = "ELM LPV (LS)")
    axs[0].plot(T_test[:, -1],label = "Test Signal")
    axs[1].plot(Ts*np.arange(simtime),u_test)
    axs[2].plot(Ts*np.arange(simtime),p_test[0,:])
    axs[3].plot(Ts*np.arange(simtime),p_test[1,:])
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    plt.xlabel("time (s)")
    axs[0].set_ylabel("$T$")
    axs[1].set_ylabel("$u$")
    axs[2].set_ylabel("$p_1$")
    axs[3].set_ylabel("$p_2$")
    axs[0].legend()
    plt.show()