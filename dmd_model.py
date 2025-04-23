import numpy as np
import matplotlib.pyplot as plt


def ct(A):
    
    return A.conj().T
class DMDModel:

    def __init__(self, X, Y, ec=999, r=0):

        self.U, self.s, Vh = np.linalg.svd(X, full_matrices=False)
        self.V = ct(Vh)
        if r != 0:
            self.rank = r
        if ec < 1 and ec > 0:
            energy_cutoff = ec
            reduced_size = 0
            total_energy = np.sum(self.s)
            contribution = 0
            m = 0
            for i in range(self.s.shape[0]):
                contribution = contribution + self.s[i]/total_energy
                print(contribution)
                reduced_size += 1
                if contribution > (1.0-energy_cutoff):
                    break

            self.rank = reduced_size

        self.Ur = self.U[:, :self.rank]
        self.Vr = self.V[:, :self.rank]
        self.sr_inverse = np.diag(1/self.s[:self.rank])

        self.Ar = ct(self.Ur)@Y@self.Vr@self.sr_inverse

        self.eigvals, self.small_psi = np.linalg.eig(self.Ar)
        
        self.A_test = self.Ur@self.Ar@ct(self.Ur)

        self.big_psi = Y@self.Vr@self.sr_inverse@self.small_psi
        #self.big_psi2 = self.Ur@self.small_psi
        
        self.big_psi_inv = np.linalg.pinv(self.big_psi)

        #self.A_recovered = self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
        #self.A_recovered2 = self.big_psi2@np.diag(self.eigvals)@np.linalg.pinv(self.big_psi2)
        
        
    def calculate_arbitrary_state(self,x0,t):
        
        return self.big_psi@np.diag(self.eigvals**t)@self.big_psi_inv@x0
    
    
    def recover_full_matrix(self):
        
        return self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
    
    

class DMDCModel:
    
    def __init__(self, X, U, Y, ec=999, r=0):

        input_matrix = np.vstack([X,U])
        self.U_in, self.s_in, Vh = np.linalg.svd(input_matrix, full_matrices=False)
        self.V_in = ct(Vh)
        
        self.U_in1 = self.U_in[:X.shape[0],:]
        self.U_in2 = self.U_in[X.shape[0]:,:]
        if r != 0:
            self.rank = r
        if ec < 1 and ec > 0:
            energy_cutoff = ec
            reduced_size = 0
            total_energy = np.sum(self.s)
            contribution = 0
            m = 0
            for i in range(self.s.shape[0]):
                contribution = contribution + self.s[i]/total_energy
                print(contribution)
                reduced_size += 1
                if contribution > (1.0-energy_cutoff):
                    break

            self.rank = reduced_size
            
        self.U,self.s,self.V = np.linalg.svd(Y,full_matrices=False)
        self.Ur = self.U[:, :self.rank]
        self.Vr = self.V[:, :self.rank]
        self.sr_inverse = np.diag(1/self.s_in[:self.rank])
        
        
        self.Ur_in1 = self.U_in1[:, :self.rank]
        self.Ur_in2 = self.U_in2[:, :self.rank]
        self.Vr_in = self.V_in[:,:self.rank]

        self.Ar = ct(self.Ur)@Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur
        self.Br = ct(self.Ur)@Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in2)

        self.eigvals, self.small_psi = np.linalg.eig(self.Ar)
        
        self.A_test = self.Ur@self.Ar@ct(self.Ur)

        self.big_psi = Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur@self.small_psi
        #self.big_psi2 = self.Ur@self.small_psi
        
        self.big_psi_inv = np.linalg.pinv(self.big_psi)
        
        self.first_run = True

        #self.A_recovered = self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
        #self.A_recovered2 = self.big_psi2@np.diag(self.eigvals)@np.linalg.pinv(self.big_psi2)
        
        
    def update(self,x,u):
        
        if self.first_run:
            print("chupa")
            self.A = self.compute_a()
            self.B = self.compute_b()
            self.first_run = False
        
        return self.A@x + self.B@u
    
    
    def compute_a(self):
        
        return self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
    
    def compute_b(self):
        
        return self.Ur@self.Br


class DMDCModelReg:
    
    def __init__(self, X, U, Y,reg = 0, ec=999, r=0,bias = False):
        
        train_size = X.shape[1]
        input_matrix = np.vstack([X,U])
        self.bias = bias
        if bias:
            input_matrix = np.vstack([input_matrix,np.ones([1,train_size])])
        print("begin first SVD")
        self.U_in, self.s_in, Vh = np.linalg.svd(input_matrix, full_matrices=False)
        self.V_in = ct(Vh)
        
        self.U_in1 = self.U_in[:X.shape[0],:]
        self.U_in2 = self.U_in[X.shape[0]:,:]
        if r != 0:
            self.rank = r
        if ec < 1 and ec > 0:
            energy_cutoff = ec
            reduced_size = 0
            total_energy = np.sum(self.s)
            contribution = 0
            m = 0
            for i in range(self.s.shape[0]):
                contribution = contribution + self.s[i]/total_energy
                print(contribution)
                reduced_size += 1
                if contribution > (1.0-energy_cutoff):
                    break

            self.rank = reduced_size
        print("begin second SVD")    
        self.U,self.s,self.V = np.linalg.svd(Y,full_matrices=False)
        print("end second svd")
        self.Ur = self.U[:, :self.rank]
        self.Vr = self.V[:, :self.rank]
        self.sr_inverse = np.diag(self.s_in[:self.rank]/(self.s_in[:self.rank]**2 + reg**2))
        
        
        self.Ur_in1 = self.U_in1[:, :self.rank]
        self.Ur_in2 = self.U_in2[:, :self.rank]
        self.Vr_in = self.V_in[:,:self.rank]
        print("begin mounting the reduced matrices")
        self.Ar = ct(self.Ur)@Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur
        self.Br = ct(self.Ur)@Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in2)
        print("end mounting the reduced matrices")

        self.eigvals, self.small_psi = np.linalg.eig(self.Ar)
        
        self.A_test = self.Ur@self.Ar@ct(self.Ur)

        self.big_psi = Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur@self.small_psi
        #self.big_psi2 = self.Ur@self.small_psi
        
        self.big_psi_inv = np.linalg.pinv(self.big_psi)
        
        self.first_run = True

        #self.A_recovered = self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
        #self.A_recovered2 = self.big_psi2@np.diag(self.eigvals)@np.linalg.pinv(self.big_psi2)
        
        
    def update(self,x,u):
        
        if self.first_run:
            print("chupa")
            self.A = self.compute_a()
            self.B = self.compute_b()
            self.first_run = False
        
        u_bias = u
        
        if self.bias:
            u_bias = np.vstack([u,np.ones([1,1])])
        return self.A@x + self.B@u_bias
    
    
    def compute_a(self):
        
        return self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
    
    def compute_b(self):
        
        return self.Ur@self.Br
    
    
    
    
class DMDioModelReg:
    
    def __init__(self, X, U, Xnext, Y,x0,reg = 0, ec=999, r=0,bias = False):
        
        train_size = X.shape[1]
        state_size = X.shape[0]
        self.x = x0
        input_matrix = np.vstack([X,U])
        self.bias = bias
        if bias:
            input_matrix = np.vstack([input_matrix,np.ones([1,train_size])])
        self.U_in, self.s_in, Vh = np.linalg.svd(input_matrix, full_matrices=False)
        self.V_in = ct(Vh)
        
        self.U_in1 = self.U_in[:X.shape[0],:]
        self.U_in2 = self.U_in[X.shape[0]:,:]
        if r != 0:
            self.rank = r
        if ec < 1 and ec > 0:
            energy_cutoff = ec
            reduced_size = 0
            total_energy = np.sum(self.s)
            contribution = 0
            m = 0
            for i in range(self.s.shape[0]):
                contribution = contribution + self.s[i]/total_energy
                print(contribution)
                reduced_size += 1
                if contribution > (1.0-energy_cutoff):
                    break

            self.rank = reduced_size
            
        self.U,self.s,self.V = np.linalg.svd(Xnext,full_matrices=False)
        self.Ur = self.U[:, :self.rank]
        self.Vr = self.V[:, :self.rank]
        self.sr_inverse = np.diag(self.s_in[:self.rank]/(self.s_in[:self.rank]**2 + reg**2))
        
        
        self.Ur_in1 = self.U_in1[:, :self.rank]
        self.Ur_in2 = self.U_in2[:, :self.rank]
        self.Vr_in = self.V_in[:,:self.rank]

        self.Ar = ct(self.Ur)@Xnext@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur
        self.Br = ct(self.Ur)@Xnext@self.Vr_in@self.sr_inverse@ct(self.Ur_in2)
        self.Cr = Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur
        self.Dr = Y@self.Vr_in@self.sr_inverse@ct(self.Ur_in2)

        self.eigvals, self.small_psi = np.linalg.eig(self.Ar)
        
        self.A_test = self.Ur@self.Ar@ct(self.Ur)

        self.big_psi = Xnext@self.Vr_in@self.sr_inverse@ct(self.Ur_in1)@self.Ur@self.small_psi
        #self.big_psi2 = self.Ur@self.small_psi
        
        self.big_psi_inv = np.linalg.pinv(self.big_psi)
        
        self.first_run = True
        self.z = ct(self.Ur)@self.x
        #self.A_recovered = self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
        #self.A_recovered2 = self.big_psi2@np.diag(self.eigvals)@np.linalg.pinv(self.big_psi2)
        
    
        
    
    def update_rom(self,u):
        u_bias = u
        
        if self.bias:
            u_bias = np.vstack([u,np.ones([1,1])])
        
        self.z = self.Ar@self.z + self.Br@u_bias
        return self.Cr@self.z + self.Dr@u_bias
    
    def update(self,u):
        
        if self.first_run:
            print("chupa")
            self.A = self.compute_a()
            self.B = self.compute_b()
            self.C = self.compute_c()
            self.D = self.Dr
            self.first_run = False
        
        u_bias = u
        
        if self.bias:
            u_bias = np.vstack([u,np.ones([1,1])])
        self.x = self.A@self.x + self.B@u_bias
        return self.C@self.x + self.D@u_bias
    
    def get_state(self):
        
        return self.x
    
    def compute_a(self):
        
        return self.big_psi@np.diag(self.eigvals)@self.big_psi_inv
    
    def compute_b(self):
        
        return self.Ur@self.Br
    
    def compute_c(self):
        
        return self.Cr@ct(self.Ur)

class piDMDTridiagModel:
    def __init__(self,X,Y,reg = 1e-5):
        
        state_dim = X.shape[0]
        training_size = X.shape[1]
        
        tridiagonal_list = []
        
        for i in range(state_dim):
            if i == 0:
                cur_X = np.vstack([np.zeros([1,training_size]),X[i:i+2]])
            elif i == state_dim - 1:
                cur_X = np.vstack([X[i-1:i+1],np.zeros([1,training_size])])
            else:
                cur_X = np.copy(X[i-1:i+2])
            cur_coeffs = Y[i]@cur_X.T@np.linalg.inv(cur_X@cur_X.T + reg*np.eye(cur_X.shape[0]))
            #cur_coeffs = np.linalg.solve(cur_X@cur_X.T,Y[i]@cur_X.T)
            tridiagonal_list += [cur_coeffs]
            
        self.A = np.zeros([state_dim,state_dim])
        
        for i in range(state_dim):
            if i==0:
                self.A[i,i:i+2] = tridiagonal_list[i][1:]
            elif i== state_dim-1:
                self.A[i,i-1:i+1] = tridiagonal_list[i][:2]
            else:
                self.A[i,i-1:i+2] = tridiagonal_list[i]
                
    def update_model(self,x0):
            
        return self.A@x0
        