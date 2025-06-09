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


#T_plot_lessdata = T_plot[:T_plot.shape[0]//2]


T0 = 0*np.ones([number_of_states,1])
T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])



parameter_data = p_signal

print(parameter_data.shape)



U_svd,s,V = diffusion_black_box_model.get_svd_from_data(T_plot_init.T,u_signal.T,parameter_data.T)
    
import scipy.io as io 


svd_dict = {'U':U_svd,'s':s,'v':V}


io.savemat('svd_from_input.mat',svd_dict)
    
    

    
    
