#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:24:26 2024

@author: jeanpj
"""

#Generate Test set using linear diffusion.

from lineardiffusion_eq import *
import matplotlib.pyplot as plt
import scipy.io as io
from lpvs_ident import *

load_dict = io.loadmat(open("lineardiffusion_test.mat",'rb'))

u_test = load_dict['u_signal'].T
p_test = load_dict['p_signal'].T
T_test = load_dict['T_plot']



#Declare model structure

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

n_in = 1
n_p = 1
number_of_states = 49


#Train comissioned models. Local


local_models_list = []
local_models_list_u = []
local_models_list_o = []
local_models_list_l = []
localrank_list = [5,10,15,20]

loxal_data_dict = io.loadmat(open("lineardiffusion_data_local.mat",'rb'))


u_signal_list = loxal_data_dict['u_signal_list']
T_plot_list = loxal_data_dict['T_plot_list']

p_list = loxal_data_dict['p_list']

T0 = 0*np.ones([number_of_states,1])
T_plot_init_list = []
for i,p in enumerate(p_list):
    T_plot_init = np.hstack([T0,T_plot_list[i][:,:-1]])
    
    T_plot_init_list += [T_plot_init]


for rank in localrank_list:
    diffusion_black_box_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)
    diffusion_black_box_model_u = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun_under,k_fun_under)
    diffusion_black_box_model_o = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun_over,k_fun_over)
    diffusion_black_box_model_l = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun_linear,k_fun_linear)
    diffusion_black_box_model.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = rank)
    local_models_list += [diffusion_black_box_model]
    diffusion_black_box_model_u.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = rank)
    local_models_list_u += [diffusion_black_box_model_u]
    diffusion_black_box_model_o.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = rank)
    local_models_list_o += [diffusion_black_box_model_o]
    diffusion_black_box_model_l.local_train_alt(T_plot_init_list,u_signal_list,p_list,T_plot_list,pod_rank = rank,proc_rank = rank)
    local_models_list_l += [diffusion_black_box_model_l]


#One step prediction for test data
    T_plot_init = np.vstack([T0.T,T_test[:-1,:]])
    e = diffusion_black_box_model.get_error_from_series(T_plot_init.T, u_test.T, p_test.T, T_test.T)
    e_u = diffusion_black_box_model_u.get_error_from_series(T_plot_init.T, u_test.T, p_test.T, T_test.T)
    e_o = diffusion_black_box_model_o.get_error_from_series(T_plot_init.T, u_test.T, p_test.T, T_test.T)
    e_l = diffusion_black_box_model_l.get_error_from_series(T_plot_init.T, u_test.T, p_test.T, T_test.T)
    print(e,"one step prediction for local training with rank",rank)
    print(e_u,"one step prediction for local training with rank (underestimated)",rank)
    print(e_o,"one step prediction for local training with rank (overestimated)",rank)
    print(e_l,"one step prediction for local training with rank (no parameter)",rank)



#Train comissioned models. Global.

globalproc_list = [40,50,60]
globalpod_list = [5,10,15]

global_model_list2 = []

global_model_list2_u = []
global_model_list2_o = []
global_model_list2_l = []

from lineardata_plot import *

T0 = 0*np.ones([number_of_states,1])
T_plot_init = np.vstack([T0.T,T_plot[:-1,:]])

parameter_data = p_signal


e_list = np.empty([len(globalproc_list),len(globalpod_list)])
e_t_list = np.empty_like(e_list)

e_u_list = np.empty_like(e_list)

e_o_list = np.empty_like(e_list)

e_l_list = np.empty_like(e_list)

for i,red_order in enumerate(globalproc_list):
    global_model_list = []
    
    global_model_list_u = []
    global_model_list_o = []
    global_model_list_l = []
    
    for j,pod_rank in enumerate(globalpod_list):
        print(i,j)
        diffusion_black_box_model = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun,k_fun)
        diffusion_black_box_model_u = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun_under,k_fun_under)
        diffusion_black_box_model_o = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun_over,k_fun_over)
        diffusion_black_box_model_l = BlackBoxLPVS(n_in,n_p,number_of_states,k_fun_linear,k_fun_linear)
        
        
        e = diffusion_black_box_model.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order,rank = pod_rank)
        dum = diffusion_black_box_model_u.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order,rank = pod_rank)
        dum = diffusion_black_box_model_o.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order,rank = pod_rank)
        dum = diffusion_black_box_model_l.global_train(T_plot_init.T,u_signal.T,parameter_data.T,T_plot.T,red_order = red_order,rank = pod_rank)
        e_list[i,j] = e
        T_test_init = np.vstack([T0.T,T_test[:-1,:]])
        e_t_list[i,j] = diffusion_black_box_model.get_error_from_series(T_test_init.T, u_test.T, p_test.T, T_test.T)
        e_u_list[i,j] = diffusion_black_box_model_u.get_error_from_series(T_test_init.T, u_test.T, p_test.T, T_test.T)
        e_o_list[i,j] = diffusion_black_box_model_o.get_error_from_series(T_test_init.T, u_test.T, p_test.T, T_test.T)
        e_l_list[i,j] = diffusion_black_box_model_l.get_error_from_series(T_test_init.T, u_test.T, p_test.T, T_test.T)
        
        global_model_list += [diffusion_black_box_model]
        
        global_model_list_u += [diffusion_black_box_model_u]
        global_model_list_o += [diffusion_black_box_model_o]
        global_model_list_l += [diffusion_black_box_model_l]
    global_model_list2 += [global_model_list]
    global_model_list2_o += [global_model_list_u]
    global_model_list2_u += [global_model_list_o]
    global_model_list2_l += [global_model_list_l]
        
    
for i,red_order in enumerate(globalproc_list):
    for j,pod_rank in enumerate(globalpod_list):    
        #print(e_list[i,j],"test error with Procrustes rank = ",red_order,"POD = ",pod_rank)
        print(e_t_list[i,j],"test error with Procrustes rank = ",red_order,"POD = ",pod_rank)



#save models for later simulation.

import pickle 

models_dict = {'global_models':global_model_list2,'local_models': local_models_list}

models_dict_u = {'global_models':global_model_list2_u,'local_models': local_models_list_u}
models_dict_o = {'global_models':global_model_list2_o,'local_models': local_models_list_o}
models_dict_l = {'global_models':global_model_list2_l,'local_models': local_models_list_l}


filename = open("lineardiff_modelsfile.pickle",'wb')
pickle.dump(models_dict,filename)


filename = open("lineardiff_modelsfile_under.pickle",'wb')
pickle.dump(models_dict_u,filename)

filename = open("lineardiff_modelsfile_over.pickle",'wb')
pickle.dump(models_dict_o,filename)

filename = open("lineardiff_modelsfile_linear.pickle",'wb')
pickle.dump(models_dict_l,filename)


