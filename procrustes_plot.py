#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:49:49 2024

@author: jeanpj
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io


load_dict = io.loadmat(open("procrustes_error_wrongb.mat","rb"))
load_dict2 = io.loadmat(open("procrustes_error.mat","rb"))



red_order_list = load_dict['red_order_list']
e_list = load_dict['e_list']

red_order_list2 = load_dict2['red_order_list']
e_list2 = load_dict2['e_list']

print(red_order_list)
print(e_list)

plt.plot(red_order_list.flatten()[1:],e_list.flatten()[1:],'s',label="b=3.2")
plt.plot(red_order_list2.flatten()[1:],e_list2.flatten()[1:],'s',label="b=3.2")
plt.grid(True)
plt.xlabel("Procrustes ramk")
plt.ylabel("training mse")
plt.show()