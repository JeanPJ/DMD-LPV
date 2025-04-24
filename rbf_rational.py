#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:30:48 2025

@author: jeanjordanou
"""

import numpy as np

def k(p):
    
    
    return p[0]*p[1]/(p[0] + p[1])**2


def rbf_basis(p,Nrbf):
    
    rbf_vec = np.empty([Nrbf,1])
    
    basis1 = np.array([[0,1]])
    
    basis2 = np.array([[1,0]])
    
    
    for i in range(Nrbf):
        
        a1 = i/Nrbf
        a2 = i/Nrbf
        
        c = a1*basis1 + a2*basis2
        
        rbf_vec[i] = np.exp(-gamma*np.linalg.norm(p - c))
    return rbf_vec


def rbf(p,w):
    
    gamma = 1
    
    Nrbf = w.shape[1]
    
    
    rbf_vec = rpf_basis(p)

    return w@rbf_vec        
        
        


N_data = 100    
P = np.random.rand(2,N_data)


Nrbf = 10


    
    
    
    
    