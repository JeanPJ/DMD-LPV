#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:00:14 2024

@author: jeanpj
"""

import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np


h = 0.005
number_of_states = int(np.ceil(1/h) - 1)

filename = "lineardiffusion_data_of" + str(number_of_states) + "states.mat"
load_dict = io.loadmat(open(filename,'rb'))

#load_dict = io.loadmat(open("diffusion_data002.mat",'rb'))




T_plot = load_dict['T_plot']
u_signal = load_dict['u_signal'].T


print(T_plot.shape)
print(u_signal.shape)
simtime = T_plot.shape[0]

number_of_states = T_plot.shape[1]

Ts = 1e-3


if __name__ == '__main__':
    fig, axs = plt.subplots(2)
    fig.suptitle('Nonlinear Diffusion Equation Dataset')
    for i in range(number_of_states):
        axs[0].plot(Ts*np.arange(simtime),T_plot[:,i])
    axs[1].plot(Ts*np.arange(simtime),u_signal)
    
    axs[0].grid(True)
    axs[1].grid(True)
    plt.xlabel("time (s)")
    axs[0].set_ylabel("$T$")
    axs[1].set_ylabel("$u(t)$")
    plt.show()
    
    
    
    zoom_from = 10000
    zoom_to = 20000
    zoom_size = zoom_to - zoom_from
    fig, axs = plt.subplots(2)
    fig.suptitle('Nonlinear Diffusion Equation Dataset')
    for i in range(0,number_of_states,20):
        axs[0].plot(Ts*np.arange(zoom_from,zoom_from + zoom_size),T_plot[zoom_from:zoom_to,i])
    axs[1].plot(Ts*np.arange(zoom_from,zoom_from + zoom_size),u_signal[zoom_from:zoom_to,:])
    
    axs[0].grid(True)
    axs[1].grid(True)
    plt.xlabel("time (s)")
    axs[0].set_ylabel("$T$")
    axs[1].set_ylabel("$u(t)$")
    plt.show()