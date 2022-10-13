import numpy as np
import tensorly as tl
import tensorly.tenalg.proximal as proximal


class WDGTC_Tucker:
    def isNum(self,value):
        try:
            value + 1
        except TypeError:
            return False
        else:
            return True
        
    def __init__(self, tucker_dims, sample_dims, lmb, lmb_a=2000, max_itr=100, lmb_c=1, mu=1e-2, rho=10, max_mu=1e10, orthogonal=True, anomaly=False, tol=1e-10):
        '''
        Parameters
        ----------
        tucker_dims : list, [a,b,c,...]
            Dimensions of core tensor, the number should be the same as the original tensor on sample dimensions
        sample_dims : list, [a,b,c,...]
            Which dimensions of the original dimensions are sample dimensions with graph penalty
        lmb : list, [a,b,c,...] or number 
            Penalty on sample dimensions. If number: same penalty on all sample dimensions. If list: different penalty on different dimensions
        lmb_a : number
            Penalty term of the anomaly        
        max_itr : 
            Maximum iteration. The default is 100.
        lmb_c : number
            Penalty on the core tensor. The default is 1.
        mu : number
            Initial value of ADMM parameter. The default is 1e-2.
        rho : number 
            The number each time mu is multiplied. The default is 10.
        orthogonal : booled
            Whether the orthogonal constraints is applied. The default is True.
        tol : 
            The tolerence of loss function The default is 1e-10.
        '''
        self.tucker_dims=tucker_dims
        self.sample_dims=sample_dims
        if self.isNum(lmb):
            self.lmb=[lmb]*len(self.tucker_dims)
        else:
            self.lmb=lmb
        self.max_itr=max_itr
        self.lmb_c=lmb_c
        self.mu=mu
        self.rho=rho
        self.lmb_a=lmb_a
        self.max_mu=max_mu
        self.orthogonal=orthogonal
        self.anomaly=anomaly
        self.tol=tol
        self.recons_loss_=[]
        self.L1_loss_=[]
        self.graph_loss_=[]
        self.combined_loss_=[]
        self.L2_loss_=0
        
        
    def fit(self,X,Omega,L):
        '''
        This algorithm uses ADMM and block coordinate descent to do the optimization. 

        Parameters
        ----------
        X : np.ndarray
            Full original data
        Omega : np.ndarray of bool variable
            Binary variable of missing location; missing = 1
        L : List []
            Laplacian matrices for sample dimensions. The length of L is equal to the order of X. 
            In sample dimension index, the component of L is the Laplacian matrix of that dimension
            Otherwise, the component of L is 0

        Returns
        -------
        C : np.ndarray
            Core tensor
        U : List []
            List of factor matrices. The length of U is equal to the order of X.
            In sample dimension index, the component of U is identity matrix
            Otherwise, U is the factor matrix of that dimension
        Y : np.ndarray
            Predicted tensor
        '''
        
        # ------------Check Conditions -------- #
        if len(X.shape)!=len(self.tucker_dims):
            raise ValueError('Tensor order does not match')
        for n in self.sample_dims:
            if X.shape[n]!=self.tucker_dims[n]:
                raise ValueError('Sample dimension error')
        if len(L)!=len(self.tucker_dims):
            raise ValueError('Laplacian matrix form error')
            
        
        # --------------Initialize------------- #
        C = tl.tensor(np.random.rand(*self.tucker_dims))
        U = self.initialize_Us(X.shape)
        
        Y = X.copy()
        if self.anomaly==True:
            A=np.zeros(X.shape)
        Y[np.nonzero(Omega)] = 0
        E = self.initialize_Es()
        S=[0]*len(X.shape)
        
        # --------------Iterate---------------- #
        for itr in range(self.max_itr):
            ## ---Update Us----- ##
            for k in list(set(range(len(X.shape)))-set(self.sample_dims)):
                W=tl.unfold(tl.tenalg.multi_mode_dot(C, U,skip=k),mode=k)
                if self.orthogonal == 1:
                    if self.anomaly == True:
                        U[k]=proximal.procrustes(tl.unfold(Y-A,k).dot(W.T))
                    else:
                        U[k]=proximal.procrustes(tl.unfold(Y,k).dot(W.T))
            
            ## ---Update Sk----- ##
            for k in self.sample_dims:
                temp=self.mu*np.eye(self.tucker_dims[k])+self.lmb[k]*L[k]
                S[k]=np.linalg.inv(temp).dot(self.mu*tl.unfold(C,mode=k)+E[k])
            
            
            ## ---Update C ----- ##  
            total_num=np.prod(self.tucker_dims)
            v_sum=np.zeros(total_num)
            for k in self.sample_dims:
                v_sum+=(S[k]-E[k]/self.mu).reshape(-1,)
            if self.orthogonal == 1:
                if self.anomaly==True:
                    temp=tl.tenalg.multi_mode_dot(Y-A, U, transpose=True).reshape(-1,)
                else:
                    temp=tl.tenalg.multi_mode_dot(Y, U, transpose=True).reshape(-1,)
                c_vec=proximal.soft_thresholding((temp+self.mu/2*v_sum)/(len(self.sample_dims)*self.mu/2+1),self.lmb_c/(len(self.sample_dims)*self.mu+2))
                C=c_vec.reshape(C.shape)
                
            ## ---Update A ----- ##
            if self.anomaly==True:
                A=proximal.soft_thresholding(Y-tl.tenalg.multi_mode_dot(C, U), 2*self.lmb_a)
            
            ## ---Update Y ----- ##  
            if self.anomaly==True:
                Y_pred = (tl.tenalg.multi_mode_dot(C, U)-A)[np.nonzero(Omega)]
            else:
                Y_pred = (tl.tenalg.multi_mode_dot(C, U))[np.nonzero(Omega)]
            Y[np.nonzero(Omega)] = Y_pred.copy()
            #Y=Y*(Y>0)
            
            ## ---Update Ek ----- ##
            for k in self.sample_dims:
                E[k] += self.mu*(tl.unfold(C,k)-S[k])
            self.mu=min(self.rho*self.mu,self.max_mu)
            
            ## ---Record loss --- ##
            if self.anomaly==True:
                p1 = np.sum((Y-tl.tenalg.multi_mode_dot(C, U)-A)**2)
                self.recons_loss_.append(p1)    
                
                p2 = np.sum(self.lmb_c*(np.abs(C)))
                self.L1_loss_.append(p2)
                
                p3=0
                for k in self.sample_dims:
                    p3+=self.lmb[k]/2*np.trace(tl.unfold(C,mode=k).T.dot(L[k].dot(tl.unfold(C,mode=k))))
                self.graph_loss_.append(p3)
                
                p4 = np.sum(self.lmb_a*np.abs(A))
                self.combined_loss_.append(np.log(p1+p2+p3+p4))
                if np.abs(self.combined_loss_[itr]-self.combined_loss_[itr-1])<self.tol and itr>2:
                    break

            
            if self.anomaly==False:
                p1 = np.sum((Y-tl.tenalg.multi_mode_dot(C, U))**2)
                self.recons_loss_.append(p1)    
            
                p2 = np.sum(self.lmb_c*(np.abs(C)))
                self.L1_loss_.append(p2)
            
                p3=0
                for k in self.sample_dims:
                    p3+=self.lmb[k]/2*np.trace(tl.unfold(C,mode=k).T.dot(L[k].dot(tl.unfold(C,mode=k))))
                self.graph_loss_.append(p3)
            
                self.combined_loss_.append(np.log(p1+p2+p3))
                if np.abs(self.combined_loss_[itr]-self.combined_loss_[itr-1])<self.tol and itr>2 and np.min(self.combined_loss_)==self.combined_loss_[-1]:
                    break
        if self.anomaly==True:
            return Y, C, U, A
        else:
            return Y, C, U       
        
        
    def initialize_Us(self, X_shape):
        U = []
        for n in range(len(X_shape)):
            if n in self.sample_dims:
                # The sample dimension is not reduced, nor rotated. Therefore it must be identity matrix
                U.append(tl.tensor(np.eye(X_shape[n])))
            else:
                U.append(tl.tensor(np.random.rand(X_shape[n], self.tucker_dims[n])))
        return U
    
    def initialize_Es(self):
        E=[]
        total_num=np.prod(self.tucker_dims)
        for n in range(len(self.tucker_dims)):
            if n in self.sample_dims:
                E.append(np.random.rand(total_num).reshape(self.tucker_dims[n],total_num // self.tucker_dims[n]))
            else:
                E.append(0)        
        return E

