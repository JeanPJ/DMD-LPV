#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:54:13 2024

@author: jeanpj
"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


load_dict = io.loadmat(open('linearprocrustes_error_pod.mat','rb'))


e_list = load_dict['e_list']



red_order_list = [10,20,30,40,50,60,80,100,120]
#red_order_list = [50]

pod_rank_list = [1,5,10,15,20,25,30,35,40,45,49]


red_order_use = [40,50,60,80]







for i,red_order in enumerate(red_order_list):
    plt.yscale('log')
    
    
    if red_order in red_order_use:
    
        plt.plot(pod_rank_list,e_list[i,:],'o',label = f"Procrustes rank {red_order}")

plt.grid(True)
plt.xlabel("POD rank")
plt.ylabel("Training MSE")    
plt.legend()    
plt.show()