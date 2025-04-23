#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:27:29 2024

@author: jeanpj
"""

from data_plot import *

def obtain_ec(s,r):
    
    return np.sum(s[:r])/np.sum(s)

print(T_plot.shape)



X = T_plot.T


U,s,Vt = np.linalg.svd(X,full_matrices=False)




# plt.plot(s[:10]/np.sum(s))
# plt.show()

ec_list = []
max_rank_in_plot = 30
min_rank_in_plot = 5
for i in range(min_rank_in_plot,max_rank_in_plot):
    
    
    ec_list += [obtain_ec(s,i)]
    
    
plt.plot(np.arange(min_rank_in_plot,max_rank_in_plot),ec_list)
plt.xlabel("reduced order rank")
plt.ylabel("energy contribution")
plt.grid(True)
plt.show()