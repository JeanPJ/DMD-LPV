#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:30:48 2025

@author: jeanjordanou
"""

import numpy as np

def k(p):
    
    
    return p[0]*p[1]/(p[0] + p[1])**2


def rbf_basis(p,Nrbf,centers):
    
    gamma = 0.01
    
    rbf_vec = np.empty([Nrbf,p.shape[1]])
    
    
    
    for i in range(Nrbf):
        
        c = np.atleast_2d(centers[i]).T
        
        rbf_vec[i] = np.exp(-gamma*np.linalg.norm(p - c)**2)
    return rbf_vec


def rbf(p,w,centers):
    
    
    Nrbf = w.shape[1]
    
    
    rbf_vec = rbf_basis(p,Nrbf,centers)

    return w@rbf_vec        
        
        


N_data = 100    
P = np.random.rand(2,N_data)


Nrbf = 50

centers = P.T

rbf_out = rbf_basis(P,Nrbf,centers)

rational_out = np.atleast_2d(k(P))


U,s,VT = np.linalg.svd(rbf_out,full_matrices = False)

V = VT.T

reg = 0.0

s_inv = np.diag(s/(s**2 + reg**2))

theta = rational_out@V@s_inv@U.T 

print(np.mean(np.abs((rational_out - theta@rbf_out)/rational_out)))

    
    
    
    
    