#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:00:46 2024

@author: jeanpj
"""

from lineardata_plot import *
from lpvs_ident import *

def k_fun(p):
    
    p = p.item()
    
    
    return np.atleast_2d([1,p,p**2]).T


n_in = 1
n_p = 1



diffusion_black_box_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)


#T_plot_lessdata = T_plot[:T_plot.shape[0]//2]


T0 = 0*np.ones([number_of_states,1])
T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])



parameter_data = p_signal

print(parameter_data.shape)


e_list = []
red_order_list = [10,20,30,40,50,60,80,100,120,0]
#red_order_list = [50]

for red_order in red_order_list:
    e = diffusion_black_box_model.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order)
    e_list += [e]
    
    
    
print(e_list)

save_dict = {'red_order_list':red_order_list,'e_list':e_list}
filename = open('linearprocrustes_error_under.mat','wb')
io.savemat(filename,save_dict)


print(e)