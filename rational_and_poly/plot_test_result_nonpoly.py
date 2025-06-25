#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:56:50 2025

@author: jeanjordanou
"""
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np


T_sim = io.loadmat("test_simulation.mat")['T_sim'].T

load_dict = io.loadmat(open("nonpolydiffusion_test.mat",'rb'))

u_test = load_dict['u_signal'].T
p_test = load_dict['p_signal']
T_test = load_dict['T_plot']


simtime = u_test.shape[0]




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