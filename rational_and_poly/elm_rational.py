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
from scipy.special import factorial


#Good seeds: 21, 23454,1234 (best).

def k(p):
    
    
    return p[0]*p[1]/(p[0] + p[1])**2


def poly_basis(p,Ndeg):
    
    Npol = int(factorial(Ndeg + 2)/(2*factorial(Ndeg)))
    print(Npol)
    poly_vec = np.empty([Npol,p.shape[1]])
    
    
    
    bias = np.ones([1,p.shape[1]])
    
    #p_and_bias = np.vstack([bias,p])
    
    j1 = 0
    j2 = 0
    
    
    max_deg = Ndeg
    
    poly_vec[0] = p[0]**j1*p[1]**j2
    
    for i in range(1,Npol):
        
        
        
        j2 += 1
        if j2 > max_deg:
            max_deg -= 1
            j2 = 0
            j1 += 1
                
        poly_vec[i] = p[0]**j1*p[1]**j2
        
    return poly_vec



def elm(p,w):
    
    
    Nelm = w.shape[1]
    
    
    
    poly_vec,W = poly_basis(p,Nelm)

    return w@poly_vec 


N_data = 100    
P = np.random.rand(2,N_data)


Nelm = 5

elm_out = poly_basis(P,Nelm)

rational_out = np.atleast_2d(k(P))


U,s,VT = np.linalg.svd(elm_out,full_matrices = False)

V = VT.T

reg = 0.01

s_inv = np.diag(s/(s**2 + reg**2))

theta = rational_out@V@s_inv@U.T 

print(np.mean(np.abs((rational_out - theta@elm_out)/rational_out))*100,"%")

