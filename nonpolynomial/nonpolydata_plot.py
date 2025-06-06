#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:00:14 2024

@author: jeanpj
"""

import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np





load_dict = io.loadmat(open("nonpolydiffusion_data_of99states.mat",'rb'))


T_plot = load_dict['T_plot']
u_signal = load_dict['u_signal'].T
p_signal = load_dict['p_signal'].T

print(T_plot.shape)
print(u_signal.shape)
simtime = T_plot.shape[0]

number_of_states = T_plot.shape[1]

Ts = 1e-2


if __name__ == '__main__':
    fig, axs = plt.subplots(3)
    fig.suptitle('Linear-parametric Diffusion Equation Dataset')
    for i in range(number_of_states):
        axs[0].plot(Ts*np.arange(simtime),T_plot[:,i])
    axs[1].plot(Ts*np.arange(simtime),u_signal)
    axs[2].plot(Ts*np.arange(simtime),p_signal)
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    plt.xlabel("time (s)")
    axs[0].set_ylabel("$T$")
    axs[1].set_ylabel("$u(t)$")
    axs[2].set_ylabel("$p$")
    plt.show()
    
    
    
    zoom_from = 10000
    zoom_to = 20000
    zoom_size = zoom_to - zoom_from
    fig, axs = plt.subplots(3)
    fig.suptitle('Linear-parametric Diffusion Equation Dataset')
    for i in range(0,number_of_states,20):
        axs[0].plot(Ts*np.arange(zoom_from,zoom_from + zoom_size),T_plot[zoom_from:zoom_to,i])
    axs[1].plot(Ts*np.arange(zoom_from,zoom_from + zoom_size),u_signal[zoom_from:zoom_to,:])
    axs[2].plot(Ts*np.arange(zoom_from,zoom_from + zoom_size),p_signal[zoom_from:zoom_to,:])
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    plt.xlabel("time (s)")
    axs[0].set_ylabel("$T$")
    axs[1].set_ylabel("$u(t)$")
    axs[2].set_ylabel("$p$")
    plt.show()