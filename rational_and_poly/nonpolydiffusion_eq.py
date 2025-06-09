#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:14:50 2024

@author: jeanpj
"""

import numpy as np
import scipy.linalg as sla
import scipy.io as io


def rat_k(p):
    
    
    return p[0]*p[1]/(p[0] + p[1]+1)**2

def PRBS(min,max,num_steps,minimum_step):
    PRBS_sig = np.empty(num_steps)
    for i in range(num_steps):

        if i % minimum_step  == 0:
            p_val = np.random.rand()
            if p_val > 0.5:
                p_val = 1.0
            if p_val < 0.5:
                p_val = 0.0

        PRBS_sig[i] = min + p_val*(max-min)

    return PRBS_sig

def change_or_not(x,min_val,max_val):
    y = 0
    p_change = np.random.rand()
    if p_change < 0.5:
        y = x
    else:
        y = min_val + (max_val - min_val)*np.random.rand()
    return y

def RFRAS(min,max,num_steps,minimum_step):
    RFRAS_sig = np.empty(num_steps)
    val = min + (max - min)*np.random.rand()
    for i in range(num_steps):

        if i % minimum_step  == 0:
            val = change_or_not(val,min,max)


        RFRAS_sig[i] = val

    return RFRAS_sig



class OnedDiffusionEquation:
    
    
    def __init__(self,h,T0,Ts = 5e-4):
        
        
        self.Ts = Ts
        
        
        number_of_states = int(np.ceil(1/h) - 1)
        self.number_of_states = number_of_states
        if number_of_states <= 0:
            
            print("h too high, init will crash")
            
        self.h = h
            
        self.T = np.zeros([number_of_states,1])
        
        self.T = T0
    
        self.D1 = np.zeros([number_of_states,number_of_states])
        
        
        self.D2 = np.zeros([number_of_states,number_of_states])
        
        
        self.D1[0,:2] = np.array([0,1]) #Matrix resulting from the interaction of the first order derivative between states
        self.D2[0,:2] = np.array([-2,1]) #Matrix resulting from the interaction of the second order derivative between states
        for i in range(1,number_of_states-1):
            self.D1[i,i-1:i+2] = np.array([-1,0,1])
            self.D2[i,i-1:i+2] = np.array([1,-2,1])
        
        self.D1[-1,-2:] = np.array([-1,1])
        self.D2[-1,-2:] = np.array([1,-1])
        
        
        self.B = np.zeros([number_of_states,1]) #The input is inserted into the first boundary condition, a heat source.
        
        self.B[0] = 1 #The input is inserted into the first boundary condition, a heat source.
        
        self.first_time = True
        
    def compute_dt(self,T,u,p):
        
        w = 0.1
        
        k = rat_k(p)
        
        
        
        
        quadratic_term = k*self.D2
        A0 = - w/(2*self.h)*self.D1
        
        step_inverse = 1/self.h
        A = 1/(self.h**2)*(quadratic_term) + A0
        
        #b_coefficient = (step_inverse*((dk[0]*2*(T[1] - T[0]))/4 - k[0])+ dk[0]*u/4 - w/2 )
        b_coefficient = step_inverse*k + w/2
        B = step_inverse*self.B*b_coefficient
        #print(A)
        #print(B)
        
        dT = A@T + B*u
        
        #print(A,A0,B,k)
        
        # if self.first_time:
        #     print(np.linalg.eigvals(A))
        #     print(k)
        #     print(A)
        #     print(self.D1)
        #     print(self.D2)
        #     self.first_time = False
        
        #print(np.linalg.eigvals(A))
        return dT
    
    def model_output(self,u,p):
        number_of_runs = 20
        for i in range(number_of_runs):
            k1 = self.compute_dt(self.T,u,p)
            k2 = self.compute_dt(self.T + k1*self.Ts/2,u,p)
            k3 = self.compute_dt(self.T + k2*self.Ts/2,u,p)
            k4 = self.compute_dt(self.T + k3*self.Ts,u,p)
        
            self.T = self.T + self.Ts/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return self.T
        
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    h = 0.01
    number_of_states = int(np.ceil(1/h) - 1)
    T0 = 0*np.ones([number_of_states,1])
    diff_eq = OnedDiffusionEquation(h,T0)
    
    
    minimum_step = 2000
    minimum_step_p = 150
    simtime = minimum_step*120

    u_signal = RFRAS(0,4,simtime,minimum_step)
    n_p = 2
    p_signal = np.empty([n_p,simtime])
    
    p_signal[0] = RFRAS(0,1,simtime,minimum_step_p)
    p_signal[1] = RFRAS(0,1,simtime,minimum_step_p)
    
    T_plot = np.empty([simtime,number_of_states])
    for k in range(simtime):
        T_plot[k] = diff_eq.model_output(u_signal[k],p_signal[:,k]).flatten()
        if k%10 == 0:
            print(k)
            print(T_plot[k,0])
        
    plt.plot(T_plot[:,:])
    plt.show()

    save_dict = {'u_signal':u_signal,'p_signal':p_signal,'T_plot':T_plot}
    filename = "nonpolydiffusion_data_of" + str(number_of_states) + "states.mat"
    save_file = open(filename,'wb')
    io.savemat(save_file,save_dict)
