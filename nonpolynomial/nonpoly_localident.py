#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:14:50 2024

@author: jeanpj
"""

import numpy as np
import scipy.linalg as sla
import scipy.io as io
from nonpolydiffusion_eq import *
        
        

if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    h = 0.01
    number_of_states = int(np.ceil(1/h) - 1)
    T0 = 0*np.ones([number_of_states,1])
    diff_eq = OnedDiffusionEquation(h,T0)
    
    
    minimum_step = 2000
    simtime = minimum_step*8
    
    #p_list = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    
    p1_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    p2_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    T_plot_list = []
    
    u_signal_list = []
    
    
    for p1 in p1_list:
        
        for p2 in p2_list:
    
            T_plot = np.empty([simtime,number_of_states])
            u_signal = RFRAS(0,4,simtime,minimum_step)
            u_signal_list += [np.atleast_2d(u_signal)]
            p = np.array([p1,p2])
            for k in range(simtime):
                T_plot[k] = diff_eq.model_output(u_signal[k],p).flatten()
                if k%10 == 0:
                    print(k)
                    print(T_plot[k,0])
            plt.plot(T_plot[:,:])
            plt.show()
            T_plot_list += [T_plot.T] 
    save_dict = {'u_signal_list':u_signal_list,'p1_list':p1_list,'p2_list':p2_list,'T_plot_list':T_plot_list}
    filename = "nonpolydiffusion_data_local" + str(number_of_states) + "states.mat"
    save_file = open(filename,'wb')
    io.savemat(save_file,save_dict)