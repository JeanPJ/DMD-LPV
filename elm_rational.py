#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 18:47:00 2025

@author: jeanpj
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:30:48 2025

@author: jeanjordanou
"""

import numpy as np

np.random.seed(21)

def k(p):
    
    
    return p[0]*p[1]/(p[0] + p[1])**2


def elm_basis(p,Nelm):
    
    elm_vec = np.empty([Nelm,p.shape[1]])
    
    W = np.random.randn(Nelm,p.shape[0]+1)
    
    print(p.shape)
    
    bias = np.ones([1,p.shape[1]])
    
    print(bias.shape)
    p_and_bias = np.vstack([bias,p])
    
    
    for i in range(Nelm):
        
        elm_vec[i] = np.tanh(W[i]@p_and_bias)
    return elm_vec



def elm(p,w):
    
    
    Nelm = w.shape[1]
    
    
    
    elm_vec = elm_basis(p,Nelm)

    return w@elm_vec 


N_data = 100    
P = np.random.rand(2,N_data)


Nelm = 30

elm_out = elm_basis(P,Nelm)

rational_out = np.atleast_2d(k(P))


U,s,VT = np.linalg.svd(elm_out,full_matrices = False)

V = VT.T

reg = 0.01

s_inv = np.diag(s/(s**2 + reg**2))

theta = rational_out@V@s_inv@U.T 

print(np.mean(np.abs((rational_out - theta@elm_out)/rational_out))*100,"%")     