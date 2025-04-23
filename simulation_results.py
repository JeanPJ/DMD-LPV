#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:20:41 2024

@author: jeanpj
"""
from lineardiffusion_eq import *
import matplotlib.pyplot as plt
import scipy.io as io
from lpvs_ident import *
import pickle


def k_fun(p):
    
    
    p = p.item()
    
    return np.atleast_2d([1,p,p**2,p**3]).T

def k_fun_under(p):
    
    
    p = p.item()
    
    return np.atleast_2d([1,p,p**2]).T


def k_fun_over(p):
    
    
    p = p.item()
    
    return np.atleast_2d([1,p,p**2,p**3,p**4]).T
    
    
def k_fun_linear(p):
    
    
    p = p.item()
    
    return np.atleast_2d([1]).T


filename = open("lineardiff_modelsfile.pickle", 'rb')
models_dict = pickle.load(filename)


filename = open("lineardiff_modelsfile_under.pickle", 'rb')
models_dict_u = pickle.load(filename)
filename = open("lineardiff_modelsfile_over.pickle", 'rb')
models_dict_o = pickle.load(filename)
filename = open("lineardiff_modelsfile_linear.pickle", 'rb')
models_dict_l = pickle.load(filename)





local_models = models_dict['local_models']
global_models = models_dict['global_models']

local_models_u = models_dict_u['local_models']
global_models_u = models_dict_u['global_models']

local_models_o = models_dict_o['local_models']
global_models_o = models_dict_o['global_models']

local_models_l = models_dict_l['local_models']
global_models_l = models_dict_l['global_models']

test_dict = io.loadmat(open("lineardiffusion_test.mat", 'rb'))

u_test = test_dict['u_signal'].T
p_test = test_dict['p_signal'].T
T_test = test_dict['T_plot']


f,ax = plt.subplots(2,sharex =True)
plt.suptitle("Test Excitation Signal")
ax[0].plot(u_test)
ax[1].plot(p_test)
for i in range(2):
    ax[i].grid(True)
    
ax[0].set_ylabel("u[k]")
ax[1].set_ylabel("p[k]")
plt.xlabel("timesteps") 
plt.show()


simtime = T_test.shape[0]


T_sim_local_list = []
number_of_states = 49
for i, model in enumerate(local_models):

    T_sim = np.empty(simtime)
    T0 = np.zeros([number_of_states, 1])

    z0 = model.get_state_reduction(T0)
    z = z0
    for k in range(simtime):

        z = model.update_latent(z, u_test[k], p_test[k])
        T_sim[k] = (model.T@z).flatten()[-1]
    T_sim_local_list += [T_sim]


plt.plot(T_test[:, -1],label = "Test Signal")
local_rank_list = [5,10,15]
for i, rank in enumerate(local_rank_list):
    plt.plot(T_sim_local_list[i],label = f"local rank = {rank}")
    print(np.mean((T_sim_local_list[i] - T_test[:,-1])**2))
    

plt.xlabel("timesteps")
plt.ylabel("T for x = 0.98")
plt.legend()
plt.grid(True)
plt.show()

T_sim_global_list2 = []
for i, model_list in enumerate(global_models):
    T_sim_global_list = [] 
    
    for j,model in enumerate(model_list):

        T_sim = np.empty(simtime)
        T0 = np.zeros([number_of_states, 1])

        z0 = model.get_state_reduction(T0)
        z = z0
        for k in range(simtime):

            z = model.update_latent(z, u_test[k], p_test[k])
            T_sim[k] = (model.T@z).flatten()[-1]
        T_sim_global_list += [T_sim]
    
    T_sim_global_list2 += [T_sim_global_list]


#plt.plot(T_test[:, -1])
globalproc_list = [40,50,60]
globalpod_list = [5,10,15]

fig,ax = plt.subplots(len(globalproc_list),sharex = True)
plt.suptitle("T at x = 0.98")
for i, proc_rank in enumerate(globalproc_list):
    ax[i].set_ylabel(f"Pr rank {proc_rank}")
    ax[i].plot(T_test[:, -1],label = "Test Signal")
    ax[i].grid(True)
    for j,pod_rank in enumerate(globalpod_list):
        e = np.mean((T_sim_global_list2[i][j] - T_test[:,-1])**2)
        if e < 1:
            ax[i].plot(T_sim_global_list2[i][j],label = f"pod rank = {pod_rank}")
        print(e)
    

plt.xlabel("timesteps")
plt.legend()
plt.show()




T_sim_u_local = np.empty(simtime)
T_sim_o_local = np.empty(simtime)

T0 = np.zeros([number_of_states, 1])

z_u = local_models_u[0].get_state_reduction(T0)
z_o = local_models_o[0].get_state_reduction(T0)
for k in range(simtime):

    z_u = local_models_u[0].update_latent(z_u, u_test[k], p_test[k])
    T_sim_u_local[k] = (local_models_u[0].T@z_u).flatten()[-1]
    z_o = local_models_o[0].update_latent(z_o, u_test[k], p_test[k])
    T_sim_o_local[k] = (local_models_o[0].T@z_o).flatten()[-1]

f,ax = plt.subplots(2,sharex = True)
plt.xlabel("timesteps")    
ax[0].plot(T_test[:, -1],label = "Test Signal")
ax[0].plot(T_sim_u_local,label = "f(p,p^2)")
ax[0].plot(T_sim_o_local,label = "f(p,p^2,p^3,p^4)")
ax[0].plot(T_sim_local_list[0],label = "exact polynomial")
ax[0].set_ylabel("local")




T_sim_u_global = np.empty(simtime)
T_sim_o_global = np.empty(simtime)

T0 = np.zeros([number_of_states, 1])

z_u = global_models_u[0][0].get_state_reduction(T0)
z_o = global_models_o[0][0].get_state_reduction(T0)
for k in range(simtime):

    z_u = global_models_u[0][0].update_latent(z_u, u_test[k], p_test[k])
    T_sim_u_global[k] = (local_models_u[0].T@z_u).flatten()[-1]
    z_o = global_models_o[0][0].update_latent(z_o, u_test[k], p_test[k])
    T_sim_o_global[k] = (local_models_o[0].T@z_o).flatten()[-1]


ax[1].plot(T_test[:, -1],label = "Test Signal")
ax[1].plot(T_sim_u_global,label = "f(p,p^2)")
ax[1].plot(T_sim_o_global,label = "f(p,p^2,p^3,p^4)")
ax[1].plot(T_sim_global_list2[0][0],label = "exact polynomial")
ax[1].set_ylabel("global")


plt.legend()
ax[0].grid(True)
ax[1].grid(True)
plt.show()