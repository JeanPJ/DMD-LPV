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




def poly_basis(p,Ndeg):
    
    Npol = int(factorial(Ndeg + 2)/(2*factorial(Ndeg)))
    print(Npol)
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


#T_plot_lessdata = T_plot[:T_plot.shape[0]//2]


T0 = 0*np.ones([number_of_states,1])
T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])



parameter_data = p_signal

print(parameter_data.shape)


e_list = []
red_order_list = [110]
#red_order_list = [50]

pod_rank = 10

reg = 5e-2

for red_order in red_order_list:
    e = diffusion_black_box_model.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order,rank = pod_rank,reg = reg)
    #e = diffusion_black_box_model.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order)
    e_list += [e]
    
    
    
print(e_list)

# save_dict = {'red_order_list':red_order_list,'e_list':e_list}
# filename = open('elmprocrustes_error.mat','wb')
# io.savemat(filename,save_dict)


print(e)

load_dict = io.loadmat(open("nonpolydiffusion_test.mat",'rb'))

u_test = load_dict['u_signal'].T
p_test = load_dict['p_signal']
T_test = load_dict['T_plot']


simtime = T_test.shape[0]


T_sim = np.empty(simtime)
T0 = np.zeros([number_of_states, 1])

z0 = diffusion_black_box_model.get_state_reduction(T0)
z = z0

#z = T0

for k in range(simtime):

    z = diffusion_black_box_model.update_latent(z, u_test[k], p_test[:,k:k+1])
    T_sim[k] = (diffusion_black_box_model.T@z).flatten()[-1]
    #T_sim[k] = z.flatten()[-1]
    
plt.plot(T_sim,label = "ELM DMD-LPV")
plt.plot(T_test[:, -1],label = "Test Signal")
plt.xlabel("timesteps")
plt.ylabel("T (global)")
plt.legend()
plt.grid(True)
plt.show()
    
    

    










