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
load_dict_under = io.loadmat(open('linearprocrustes_error_pod_unrder.mat','rb'))
load_dict_over = io.loadmat(open('linearprocrustes_error_pod_over.mat','rb'))

e_list = load_dict['e_list']
e_list_under = load_dict_under['e_list']
e_list_over = load_dict_over['e_list']



red_order_list = [10,20,30,40,50,60,80,100,120]
#red_order_list = [50]

pod_rank_list = [1,5,10,15,20,25,30,35,40,45,49]


red_order_use = [50,60]




f,ax = plt.subplots(2,sharex = True)
ax[0].set_yscale('log')
ax[1].set_yscale('log')

sub_index = 0
for i,red_order in enumerate(red_order_list):
    plt.yscale('log')
    
    
    if red_order in red_order_use:
    
        ax[sub_index].plot(pod_rank_list,e_list[i,:],'o',label = f"f(p) exact")
        sub_index += 1


ax[0].plot(pod_rank_list,e_list_under[0,:],'o',label = f"f(p) underestimated")
ax[0].plot(pod_rank_list,e_list_over[0,:],'o',label = f"f(p) overestimated")

ax[1].plot(pod_rank_list,e_list_under[1,:],'o',label = f"f(p) underestimated")
ax[1].plot(pod_rank_list,e_list_over[1,:],'o',label = f"f(p) overestimated")


ax[0].grid(True)
ax[1].grid(True)
plt.xlabel("POD rank")
ax[0].set_ylabel("MSE (rank 50)") 
ax[1].set_ylabel("MSE (rank 60)")       
plt.legend()    
plt.show()