#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:58:08 2020

@author: jean-jordanou
"""
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
#objetivo: Criar uma classe que modela uma EDO qualquer em CASADI.



class SimulationModel:
    """
    Template class for an arbitrary model used to simulate the real plant to test controller.
    Should be more sophisticated than the blackboxmodel type.
    """
    
    def __init__(self):
        pass
    
    def model_output(self,u):
        """
        outputs the model given an input, should return the model output. Note that
        the states should be an internal variable.
        :param u: input to the system. Dimension: 1d array of size PNMPCController.n_mv
        :type u: numpy.ndarray
        :returns: one step prediction given u and internal state. Dimension: 1darray of size PNMPCController.n_out
        :rtype: numpy.ndarray
        """
        pass
    

class CasadiModel(SimulationModel):
    
    """
    Class for an arbitrary nonlinear ODE which uses CASADI as an engine
    """
    
    def __init__(self,f,g,x0,n_in,n_out,Ts,nonnegatives = []):
        """
        Class initialization functions f and g must be vectorized.
        :param f: state function, wih input as x and u, and output as dx. Should return an MX.
        :param g: output function, with input as x, and output as y.
        :param n_in: Number of inputs.
        :type n_in: int
        :param x0: Initial condition.
        :type n_s: int
        :param n_out: number ouf outputs.
        :type n_out: int
        """
        self.x = x0
        self.Ts = Ts
        self.t = 0.0
        self.nonnegatives = nonnegatives
        
        #symbolic variables 
        self.u_sim = MX.sym('u',n_in) #simbolic input
        self.x_sim = MX.sym('x',x0.shape[0]) #simbolic state
        self.y_sim = MX.sym('y',n_out) #symbolic output
        
        #symbolic functions
        #self.f = Function('f',[x,u],f)
        #self.g = Function('f',{x,u},g)
        
        #ode definition 
        self.ode = {'x':self.x_sim,'p':self.u_sim,'ode':f(self.x_sim,self.u_sim)}
        
        #create integrator:
        
        self.F = integrator('F','cvodes',self.ode,{'tf':Ts})
        self.g = g
        
        
    def model_output(self,u,array_form = True):
        
        run_number = 1
        
        for i in range(run_number):
            self.F = integrator('F','cvodes',self.ode,{'t0':self.t,'tf':self.t + self.Ts/run_number,'reltol':1e-6})
            sim_result = self.F(x0=self.x,p = u)
            self.result = sim_result
            self.t = self.t + self.Ts
            self.x = sim_result['xf']
            for j in self.nonnegatives:
                if self.x[j] < 1e-9:
                    self.x[j] = 1e-9
        
        y = self.g(self.x)
        
        if array_form == True:
            return np.array(y).flatten()
        else:
            return y
    
    
# def lorentz_state_function(x,u):
    
#     sigma = 10.0
#     rho = 28.0
#     beta = 8.0/3.0
    
#     dx = MX(3,1)
    
    
#     dx[0] = sigma*(x[1] - x[0])
#     dx[1] = x[0]*(rho - x[2]) - x[1]
#     dx[2] = x[0]*x[1] -beta*x[2]

    
#     return dx
        

# idfunc = lambda x:x

# x0 = np.array([-2.0,-7.0,25.0])


# Ts = 0.01
# lorentz_system = CasadiModel(lorentz_state_function,idfunc,x0,1,3,Ts)

# simtime = 1100

# y_plot = np.empty([simtime,3])
# for k in range(simtime):
#     y_plot[k] = np.array(lorentz_system.model_output(28.0)).flatten()
    
# lyap_exp = 0.934
# plt.plot(lyap_exp*0.01*np.arange(simtime),y_plot)
# plt.show()