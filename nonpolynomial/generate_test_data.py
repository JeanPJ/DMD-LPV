#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:24:26 2024

@author: jeanpj
"""

#Generate Test set using linear diffusion.

from nonpolydiffusion_eq import *
import matplotlib.pyplot as plt
from lpvs_ident import *


h = 0.01
number_of_states = int(np.ceil(1/h) - 1)
T0 = 0*np.ones([number_of_states,1])
diff_eq = OnedDiffusionEquation(h,T0)


minimum_step = 15000
minimum_step_p = 1500
simtime = minimum_step*3

u_signal = RFRAS(0,4,simtime,minimum_step)
#u_signal = np.ones(simtime)
n_p = 2

p_signal = np.empty([n_p,simtime])
p_signal[0] = RFRAS(0,1,simtime,minimum_step_p)
p_signal[1] = RFRAS(0,1,simtime,minimum_step_p)

T_plot = np.empty([simtime,number_of_states])
for k in range(simtime):
    T_plot[k] = diff_eq.model_output(u_signal[k],p_signal[:,k]).flatten()
    if k%10 == 0:
        print(k)
        print(T_plot[k,0])
    
plt.plot(T_plot[:,:])
plt.show()

save_dict = {'u_signal':u_signal,'p_signal':p_signal,'T_plot':T_plot}

save_file = open("nonpolydiffusion_test.mat",'wb')
io.savemat(save_file,save_dict)




