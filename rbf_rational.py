#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:30:48 2025

@author: jeanjordanou
"""

import numpy as np

def k(p):
    
    
    return p[0]*p[1]/(p[0] + p[1])**2



def rbf(p,w):
    
    gamma = 1
    
    Nrbf = w.shape[1]
    
    
    rbf_vec = np.empty([Nrbf,1])
    
    
    for i in range(Nrbf):
        
        rbf_vec[i] = np.exp()        
    
    
    