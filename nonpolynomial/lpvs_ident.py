#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:04:38 2024

@author: jeanpj
Identification of a state space IO  LPV system of type:
x[k+1] = A(Theta)x[k] + B(Theta)u[k]
where theta is the parameter.
"""

import numpy as np
import scipy.linalg as sla
#import scipy.special.poch as poch
#import scipy.special.factorial as factorial
#from sklearn.utils.extmath import randomized_svd
import scipy.sparse.linalg as ssla

def my_poch(z,m):
    
    
    poch_calc = 1
    
    for i in range(m):
        
        poch_calc *= z + i + 1
    
    
    
    
    return poch_calc



affine = lambda p:np.vstack([1,p])

# def polyexpand(v,d):
    
#     """
#     given a set of elements v, return the value of all monomials of degree n.
    
#     """
#     if d == 0:
#         poly = np.atleast_2d(1)
        
#     else:
#         n = v.shape[0]
#         monsize = factorial(n + d)/(factorial(n)*factorial(d))
#         poly = np.empty([monsize,1])
        
        
        
    
#     return poly


class BlackBoxLTIS:
    
    def __init__(self,n_in,n_s):
        self.n_s = n_s
        self.n_in = n_in
        self.A = np.empty([n_s,n_s])
        self.B = np.empty([n_s,n_in])
        
    def update(self,x,u):
        
        return self.A@x + self.B@u
    
    def train(self,X,U,Y,reg=0):
        
        INPUT = np.vstack([X,U])
        U_svd,s,V = sla.svd(INPUT,full_matrices = False)
        
        V = V.T
        
        
        s_reverse = s/(s**2 + reg**2)
        
        s_reverse = np.diag(s_reverse)
        
        prod = Y@V@s_reverse
        
        self.A = prod@U_svd[:self.n_s,:].T
        self.B = prod@U_svd[self.n_s:,:].T
        
        

class BlackBoxLPVS:
    
    
    def __init__(self,n_in,n_p,n_s,scheduling_fun,scheduling_fun_in):
        
        self.n_in = n_in
        self.n_p = n_p
        self.n_s = n_s
        self.sc_f = scheduling_fun
        self.sc_fin = scheduling_fun_in
        
        dummy_p = np.zeros([n_p,1])
        
        
        
        self.n_f = self.sc_f(dummy_p).shape[0]
        self.n_fin = self.sc_fin(dummy_p).shape[0]
        
        
        self.W_A = np.zeros([n_s,n_s*self.n_f])
        self.W_B = np.zeros([n_in,n_in*self.n_fin])
        self.rank = 0
        
        
        
    def update(self,x,u,p):
        
        
        if self.rank == 0:
            return self.W_A@np.kron(self.sc_f(p),x) + self.W_B@np.kron(self.sc_fin(p),u)
        
        else:
            z = self.T.T@x
            z_next = self.W_A@np.kron(self.sc_f(p),z) + self.W_B@np.kron(self.sc_fin(p),u)
            
            #Projected DMD
            return self.T@z_next
        
        
            #Exact DMD
            
            #return self.exact_T@z_next
            
    def update_latent(self,z,u,p):
        
        z_next = self.W_A@np.kron(self.sc_f(p),z) + self.W_B@np.kron(self.sc_fin(p),u)
        
        return z_next
        
    def get_state_reduction(self,x):
        
        
        return self.T.T@x
    
    def global_train(self,X,U,P,Y,reg = 0,red_order = 0,rank = 0,par_rank = 0,par_rank_in = 0):
        
        
        
        self.rank = rank
        
        if par_rank == 0 and par_rank_in == 0:
            true_X = np.empty([self.n_s*self.n_f,X.shape[1]])
            true_U = np.empty([self.n_in*self.n_fin,U.shape[1]])
            for t in range(X.shape[1]):
            
                true_X[:,t:t+1] = np.kron(self.sc_f(P[:,t:t+1]),X[:,t:t+1])
                true_U[:,t:t+1] = np.kron(self.sc_fin(P[:,t:t+1]),U[:,t:t+1])
        
        
            
            
        elif par_rank > 0 and par_rank_in == 0:
            
            features_state = np.empty([self.n_f,X.shape[1]])
            
            for t in range(X.shape[1]):
                features_state[:,t:t+1] = self.sc_f(P[:,t:t+1])
                
            Ufs,sfs,Vtfs = np.linalg.svd(features_state,full_matrices=False)
            
            self.Tpca = Ufs[:,:par_rank]
            
            features_state_reduced = self.Tpca.T@features_state
            
            
            true_X = np.empty([self.n_s*par_rank,X.shape[1]])
            true_U = np.empty([self.n_in*self.n_fin,U.shape[1]])
            
            for t in range(X.shape[1]):
            
                true_X[:,t:t+1] = np.kron(features_state_reduced[:,t:t+1],X[:,t:t+1])
                true_U[:,t:t+1] = np.kron(self.sc_fin(P[:,t:t+1]),U[:,t:t+1])
            
            
            
        elif par_rank == 0 and par_rank_in > 0:
            features_input = np.empty([self.n_fin,X.shape[1]])
            
            for t in range(X.shape[1]):
                features_input[:,t:t+1] = self.sc_fin(P[:,t:t+1])
                
            Ufin,sfin,Vtfin = np.linalg.svd(features_input,full_matrices=False)
            
            self.Tpca_in = Ufin[:,:par_rank]
            
            features_input_reduced = self.Tpca_in.T@features_input
            
            
            true_X = np.empty([self.n_s*self.n_f,X.shape[1]])
            true_U = np.empty([self.n_in*par_rank_in,U.shape[1]])
            
            for t in range(X.shape[1]):
            
                true_X[:,t:t+1] = np.kron(self.sc_f(P[:,t:t+1]),X[:,t:t+1])
                true_U[:,t:t+1] = np.kron(features_input_reduced[:,t:t+1],U[:,t:t+1])
        else:
            features_state = np.empty([self.n_f,X.shape[1]])
            features_input = np.empty([self.n_fin,X.shape[1]])
            
            for t in range(X.shape[1]):
                features_state[:,t:t+1] = self.sc_f(P[:,t:t+1])
                features_input[:,t:t+1] = self.sc_fin(P[:,t:t+1])
                
            Ufs,sfs,Vtfs = np.linalg.svd(features_state,full_matrices=False)
            
            self.Tpca = Ufs[:,:par_rank]
            
            features_state_reduced = self.Tpca.T@features_state
            
            Ufin,sfin,Vtfin = np.linalg.svd(features_input,full_matrices=False)
            
            self.Tpca_in = Ufin[:,:par_rank]
            
            features_input_reduced = self.Tpca_in.T@features_input
            
            true_X = np.empty([self.n_s*par_rank,X.shape[1]])
            true_U = np.empty([self.n_in*par_rank_in,U.shape[1]])
            
            for t in range(X.shape[1]):
            
                true_X[:,t:t+1] = np.kron(features_state_reduced[:,t:t+1],X[:,t:t+1])
                true_U[:,t:t+1] = np.kron(features_input_reduced[:,t:t+1],U[:,t:t+1])
            
        
        INPUT = np.vstack([true_X,true_U])    
        
        #U_svd, s, V  = sla.svd(INPUT,full_matrices = False)
        
        
        
        
        if red_order == 0: #full order model case, least squares is fully solved
            U_svd, s, V  = sla.svd(INPUT,full_matrices = False)
            self.s_aug = s
        
            V = V.T
            s_reverse = s/(s**2 + reg**2)
        
        
            s_reverse = np.diag(s_reverse)
        
            prod = Y@V@s_reverse
            
            if par_rank == 0:
        
                self.W_A = prod@U_svd[:self.n_f*self.n_s,:].T
                self.W_B = prod@U_svd[self.n_f*self.n_s:,:].T
                
            else:
                
                self.W_A = prod@U_svd[:self.n_s*par_rank,:].T
                self.W_B = prod@U_svd[self.n_s*par_rank:,:].T
            
            
            return np.mean((self.W_A@true_X + self.W_B@true_U - Y)**2)
            
            
            
        elif rank == 0: #full order model case, Procrustes problem with rank(A) = red_order being solved.
            
            print("begin svd calc")
            U_svd, s, VT  = sla.svd(INPUT,full_matrices = False)
            #U_svd, s, VT  = ssla.svds(INPUT,k=red_order)
            self.s_aug = s
            print("end svd calc")
            
            
            
            #red_order_ls = red_order*self.n_f
            
            
            red_order_ls = red_order
            V = VT.T
            s_reverse = s/(s**2 + reg**2)
        
        
            Sigma_r = np.diag(s_reverse[:red_order_ls]) #pruning the singvals
        
            prod = Y@V[:,:red_order_ls]@Sigma_r
            
            if par_rank == 0:
        
                self.W_A = prod@U_svd[:self.n_f*self.n_s,:red_order_ls].T
                self.W_B = prod@U_svd[self.n_f*self.n_s:,:red_order_ls].T
                
            else:
                
                self.W_A = prod@U_svd[:self.n_s*par_rank,:red_order_ls].T
                self.W_B = prod@U_svd[self.n_s*par_rank:,:red_order_ls].T
            
            print(U_svd.shape,s.shape,V.shape)
            
            
            squared_error = (self.W_A@true_X + self.W_B@true_U - Y)**2
            
            mean_squared_errors = np.mean(squared_error,axis = 1)
            
            print(mean_squared_errors.shape)
            
            mean_of_means = np.mean(mean_squared_errors)
            
            return np.mean((self.W_A@true_X + self.W_B@true_U - Y)**2)
            
            
            
            
        else: #Procrustes problem with ROM.
            
            self.U_svd, s, VT  = sla.svd(INPUT,full_matrices = False)
            print("begin svd calc")
            #self.U_svd, s, VT  = ssla.svds(INPUT,k=red_order)
            print("end svd calc")
            #print("The SVD is being fully performed, therefore there will be no reduction in training time")
            
            
            
            #red_order_ls = red_order*self.n_f
            red_order_ls = red_order
            self.red_order_ls = red_order_ls
            V = VT.T
            s_reverse = s/(s**2 + reg**2)
        
        
            Sigma_r = np.diag(s_reverse[:red_order_ls])
            
            
            print("The Y SVD is being fully performed, therefore there will be no reduction in training time")
            self.Ut,St,VtT = sla.svd(Y,full_matrices = False)
            #Ut,St,VtT = ssla.svds(Y,k=rank)
            #Ut,St,VtT = ssla.svds(Y,k=rank,return_singular_vectors='u')
            
            #Vt = VtT.T
            
            self.T = self.Ut[:,:self.rank]
            
            bigT = np.kron(np.eye(self.n_f),self.T)
            
        
            self.prod = Y@V[:,:red_order_ls]@Sigma_r
            
            
            if par_rank == 0:
        
                self.W_A = self.T.T@self.prod@self.U_svd[:self.n_f*self.n_s,:red_order_ls].T@bigT
                self.W_B = self.T.T@self.prod@self.U_svd[self.n_f*self.n_s:,:red_order_ls].T
                
            else:
                
                self.W_A = self.T.T@self.prod@self.U_svd[:par_rank*self.n_s,:red_order_ls].T@bigT
                self.W_B = self.T.T@self.prod@self.U_svd[par_rank*self.n_s:,:red_order_ls].T
            
            
            true_Z = bigT.T@true_X
            
            
            mse_of_pod = 0
            
            for i in range(true_Z.shape[1]):
                
                mse_of_pod += np.mean((self.W_A@true_Z[:,i:i+1] + self.W_B@true_U[:,i:i+1] - self.T.T@Y[:,i:i+1])**2)
                
            mse_of_pod /= Y.shape[1]
            
            return mse_of_pod
        
    
    def retrain_procrustes(self,X,U,P,Y,red_order):
        pass
        
    
    def retrain_POD(self,X,U,P,Y,rank):
        
        
        self.rank = rank
        
        self.T = self.Ut[:,:self.rank]
        
        bigT = np.kron(np.eye(self.n_f),self.T)
    
        self.W_A = self.T.T@self.prod@self.U_svd[:self.n_f*self.n_s,:self.red_order_ls].T@bigT
        self.W_B = self.T.T@self.prod@self.U_svd[self.n_f*self.n_s:,:self.red_order_ls].T
        
        mse_of_pod = 0
        
        
        Z = self.T.T@X
        for i in range(X.shape[1]):
            
            mse_of_pod += np.mean((self.W_A@np.kron(self.sc_f(P[:,i:i+1]),Z[:,i:i+1]) + self.W_B@np.kron(self.sc_fin(P[:,i:i+1]),U[:,i:i+1]) - self.T.T@Y[:,i:i+1])**2)
            
        mse_of_pod /= Y.shape[1]
        
        return mse_of_pod
        
    
    def local_train(self,X_list,U_list,p_list,Y_list,reg = 0,pod_rank = 0,proc_rank = 0):
        
        self.rank = pod_rank
        #identifies each LT system
        LTI_list = []
        
        if pod_rank == 0:
            
            self.T = np.eye(self.n_s)
        else:
            
            Y = Y_list[0]
            
            
            for Yadd in Y_list[1:]:
                
                Y = np.hstack([Y,Yadd])
            
            U,s,V = np.linalg.svd(Y,full_matrices = False)
            
            self.T = self.T = U[:,:pod_rank]
        
        for i,p in enumerate(p_list):
            
            dummy_LTI = BlackBoxLTIS(self.n_in, self.n_s)
            
            dummy_LTI.train(X_list[i], U_list[i], Y_list[i])
            
            LTI_list += [dummy_LTI]
        
        
        if pod_rank == 0:
            ident_size = self.n_s
            
        else:
            ident_size = pod_rank
        
        
        INPUT_A = np.kron(self.sc_f(p_list[:,0:1]),np.eye(ident_size))
        INPUT_B = np.kron(self.sc_fin(p_list[:,0:1]),np.eye(self.n_in))
        
        OUTPUT_A = self.T.T@LTI_list[0].A@self.T
        OUTPUT_B = self.T.T@LTI_list[0].B
        for n,LTI in enumerate(LTI_list):
            
            if n == 0:
                pass
            else:
                OUTPUT_A = np.hstack([OUTPUT_A,LTI_list[n].A])
                OUTPUT_B = np.hstack([OUTPUT_B,LTI_list[n].B])
                
                
                INPUT_A = np.hstack([INPUT_A,np.kron(self.sc_f(p_list[:,n:n+1]),np.eye(ident_size))])
                INPUT_B = np.hstack([INPUT_B,np.kron(self.sc_fin(p_list[:,n:n+1]),np.eye(self.n_in))])
                
                
        
        
        U_svd_A, s_A, V_A  = sla.svd(INPUT_A,full_matrices = False)
        
        V_A = V_A.T
        
        s_reverse_A = s_A/(s_A**2 + reg**2)
        
        
        U_svd_B, s_B, V_B  = sla.svd(INPUT_B,full_matrices = False)
        
        V_B = V_B.T
        
        s_reverse_B = s_B/(s_B**2 + reg**2)
        
        
        if proc_rank == 0:
        
            self.W_A = OUTPUT_A@V_A@np.diag(s_reverse_A)@U_svd_A.T
            self.W_B = OUTPUT_B@V_B@np.diag(s_reverse_B)@U_svd_B.T
            
        else:
            Sigma_rA = np.diag(s_reverse_A[:proc_rank])
            
            self.W_A = OUTPUT_A@V_A[:,:proc_rank]@Sigma_rA@U_svd_A[:,:proc_rank].T
            self.W_B = OUTPUT_B@V_B@np.diag(s_reverse_B)@U_svd_B.T
            
    def local_train_alt(self,X_list,U_list,p_list,Y_list,reg = 0,pod_rank = 0,proc_rank = 0):
        
        self.rank = pod_rank
        #identifies each LT system
        LTI_list = []
        
        if pod_rank == 0:
            
            self.T = np.eye(self.n_s)
        else:
            
            Y = Y_list[0]
            
            
            for Yadd in Y_list[1:]:
                
                Y = np.hstack([Y,Yadd])
            
            U,s,V = np.linalg.svd(Y,full_matrices = False)
            
            self.T = self.T = U[:,:pod_rank]
        
        
        if pod_rank == 0:
            number_of_states = self.n_s
            
        else:
            number_of_states = pod_rank
        for i,p in enumerate(p_list):
            
            dummy_LTI = BlackBoxLTIS(self.n_in, number_of_states)
            
            dummy_LTI.train(self.T.T@X_list[i], U_list[i], self.T.T@Y_list[i])
            
            LTI_list += [dummy_LTI]
        
        
        if pod_rank == 0:
            ident_size = self.n_s
            
        else:
            ident_size = pod_rank
        
        
        INPUT_A = np.kron(self.sc_f(p_list[:,0:1]),np.eye(ident_size))
        INPUT_B = np.kron(self.sc_fin(p_list[:,0:1]),np.eye(self.n_in))
        
        OUTPUT_A = LTI_list[0].A
        OUTPUT_B = LTI_list[0].B
        for n,LTI in enumerate(LTI_list):
            
            if n == 0:
                pass
            else:
                OUTPUT_A = np.hstack([OUTPUT_A,LTI_list[n].A])
                OUTPUT_B = np.hstack([OUTPUT_B,LTI_list[n].B])
                
                
                INPUT_A = np.hstack([INPUT_A,np.kron(self.sc_f(p_list[:,n:n+1]),np.eye(ident_size))])
                INPUT_B = np.hstack([INPUT_B,np.kron(self.sc_fin(p_list[:,n:n+1]),np.eye(self.n_in))])
                
                
        
        
        U_svd_A, s_A, V_A  = sla.svd(INPUT_A,full_matrices = False)
        
        V_A = V_A.T
        
        s_reverse_A = s_A/(s_A**2 + reg**2)
        
        
        U_svd_B, s_B, V_B  = sla.svd(INPUT_B,full_matrices = False)
        
        V_B = V_B.T
        
        s_reverse_B = s_B/(s_B**2 + reg**2)
        
        
        if proc_rank == 0:
        
            self.W_A = OUTPUT_A@V_A@np.diag(s_reverse_A)@U_svd_A.T
            self.W_B = OUTPUT_B@V_B@np.diag(s_reverse_B)@U_svd_B.T
            
        else:
            Sigma_rA = np.diag(s_reverse_A[:proc_rank])
            
            self.W_A = OUTPUT_A@V_A[:,:proc_rank]@Sigma_rA@U_svd_A[:,:proc_rank].T
            self.W_B = OUTPUT_B@V_B@np.diag(s_reverse_B)@U_svd_B.T
        
        
    def get_error_from_series(self,X,U,P,Y):
        
        true_X = np.empty([self.n_s*self.n_f,X.shape[1]])
        true_U = np.empty([self.n_in*self.n_fin,U.shape[1]])
        for t in range(X.shape[1]):
        
            true_X[:,t:t+1] = np.kron(self.sc_f(P[:,t:t+1]),X[:,t:t+1])
            true_U[:,t:t+1] = np.kron(self.sc_fin(P[:,t:t+1]),U[:,t:t+1])
        
        
        if self.T.shape[1] == self.T.shape[0] or not hasattr(self,'T'):
        
            return np.mean((self.W_A@true_X + self.W_B@true_U - Y)**2)
        else:
            bigT = np.kron(np.eye(self.n_f),self.T)
            true_Z = bigT.T@true_X
            return np.mean((self.W_A@true_Z + self.W_B@true_U - self.T.T@Y)**2)
        
    def analyze_singular_values(self,X,U,P):
            
        true_X = np.empty([self.n_s*self.n_f,X.shape[1]])
        true_U = np.empty([self.n_in*self.n_f,U.shape[1]])
        
        
        for t in range(X.shape[1]):
            true_X[:,t:t+1] = np.kron(self.sc_f(P[:,t:t+1]),X[:,t:t+1])
            true_U[:,t:t+1] = np.kron(self.sc_f(P[:,t:t+1]),U[:,t:t+1])
            
        INPUT = np.vstack([true_X,true_U])
        
        print(INPUT.shape,"shape da matriz de entrada do LS")
        
        U_svd, s, V  = sla.svd(INPUT,full_matrices = False)
        
        
        
        return s/np.sum(s)
        
        
    
