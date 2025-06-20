#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:00:46 2024

@author: jeanpj
"""

from nonpolydata_plot import *
from lpvs_ident import *

import scipy.io as io
import numpy as np
from scipy.special import factorial

w_elm = io.loadmat("elm_weights.mat")['W']

svd_dict = io.loadmat("svd_from_input.mat")

U_svd = svd_dict['U']
s = svd_dict['s'].flatten()
V = svd_dict['v']




def poly_basis(p,Ndeg):
    
    Npol = int(factorial(Ndeg + 2)/(2*factorial(Ndeg)))
    poly_vec = np.empty([Npol,p.shape[1]])
    
    
    
    bias = np.ones([1,p.shape[1]])
    
    #p_and_bias = np.vstack([bias,p])
    
    j1 = 0
    j2 = 0
    
    
    max_deg = Ndeg
    
    poly_vec[0] = p[0]**j1*p[1]**j2
    
    for i in range(1,Npol):
        
        
        
        j2 += 1
        if j2 > max_deg:
            max_deg -= 1
            j2 = 0
            j1 += 1
                
        poly_vec[i] = p[0]**j1*p[1]**j2
        
    return poly_vec


n_in = 1
n_p = 2

Nelm = 5

k_fun = lambda p:poly_basis(p,Nelm)

diffusion_black_box_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)

golden_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)


#T_plot_lessdata = T_plot[:T_plot.shape[0]//2]


T0 = 0*np.ones([number_of_states,1])
T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])



parameter_data = p_signal

print(parameter_data.shape)


e_list = []
red_order_list = [110]
#red_order_list = [50]

pod_rank = [5]

#reg = 1e-1 esse deu certo

reg = 5e-2

for red_order in red_order_list:
    for pod in pod_rank:
        diffusion_black_box_model.train_from_svd(U_svd,s,V,T_plot.T,red_order = red_order,rank = pod,reg = reg)
        golden_model.train_from_svd(U_svd,s,V,T_plot.T,red_order = 0,rank = 0,reg = reg)
        e = diffusion_black_box_model.get_error_from_series(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T)
        e_golden = golden_model.get_error_from_series(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T)
        print(e,e_golden, "training (reduced, LS)")
        #e = diffusion_black_box_model.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order)
        
        
        
    
    
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
        
        z_ls = T0
        
        
        T_test_init = np.vstack([T0.T,T_test[:-1,:]])
        
        
        e_test = diffusion_black_box_model.get_error_from_series(T_test_init.T,u_test.T,p_test,T_test.T)
        e_ls = golden_model.get_error_from_series(T_test_init.T,u_test.T,p_test,T_test.T)
        print("test error:",e_test,e_ls)
        
        for k in range(simtime):
        
            z = diffusion_black_box_model.update_latent(z, u_test[k], p_test[:,k:k+1])
            z_ls = golden_model.update_latent(z_ls, u_test[k], p_test[:,k:k+1])
            T_sim[k] = (diffusion_black_box_model.T@z).flatten()[-1]
            T_ls[k] = z_ls.flatten()[-1]
            
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
        fig.suptitle('ELM DMD-LPV Simulation over Test Data')
        axs[0].plot(T_sim,label = f"ELM DMD-LPV POD rank = {pod} Pr = {red_order}")
        axs[0].plot(T_ls,label = "ELM LPV (LS)")
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
        
    
