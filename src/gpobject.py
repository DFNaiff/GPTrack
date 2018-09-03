# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import scipy.optimize as spopt

from . import utils
from . import utilsla


class GPObject(object):
    def __init__(self,kernel,noise_kernel,phi,data = None):
        self._initialize_kernels(kernel,noise_kernel,phi)
        if data != None:
            self.change_data(data)
        else:
            self._initialize_empty()
    
    def predict(self,x):
        """
            Calculate mean(x),var(x)
        """
        # TODO : add efficient update of r
        kx = utils.relation_array(self.cov,x,self.xdata)
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        s = utilsla.invumatmul(self.U,kx,trans='T')
        mean = self.m + np.dot(r,s)
        var = self.cov(x,x) - np.dot(s,s)
        return mean,var
    
    def predict_batch(self,xs):
        Kx = utils.binary_function_matrix_2(self.cov,xs,self.xdata)
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        S = utilsla.invumatmul(self.U,Kx.transpose(),trans='T')
        mean = self.m + np.dot(S.transpose(),r)
        Kxx = utils.binary_function_matrix(self.cov,xs)
        var = Kxx - np.dot(S.transpose(),S)
        return mean,var
    
    def change_data(self,data):
        """
            Changes the data of the GP, replacing it with new_data
        """
        # TODO : assertions
        x,z = data
        self.xdata = copy.copy(x)
        self.zdata = np.array(z)
        self.numdata = len(x)
        self.K = utils.binary_function_matrix(self.cov,self.xdata)
        if self.noisekernel.is_diagonal: #K(X,X) + sigma2*I
            I = np.diag([self.noisecov(xx,xx) for xx in self.xdata])
            self.K = self.K + I
        else:
            raise NotImplementedError
        self.U = utilsla.spla.cholesky(self.K,lower=False)
        self._update_likelihood()
        self.is_empty = False
        
    def downdate(self,i=0):
        # TODO : do not simply downdate likelihood
        self.K = utilsla.contract(self.K,i)
        self.U = utilsla.contract_cholesky(self.U,i)
        self.xdata = self.xdata[:i] + self.xdata[i+1:]
        # TODO : surely there's a function for dropping
        self.zdata = np.concatenate([self.zdata[:i],
                                     self.zdata[i+1:]])
        self._update_likelihood()
    
    def downdate_batch(self,drop_inds,is_sorted = True):
        # TODO : not clear whether there is a simple and more efficient way
        #        to downdate. But think of one
        # Assumes drop_inds to be sorted
        if not is_sorted:
            drop_inds = sorted(drop_inds)
        for ind in drop_inds[::-1]:
            self.downdate(ind)
            
    def update(self,new_data):
        """
            Updates data, covariance matrix and it's cholesky factor, 
            and likelihood
        """
        x_t,z_t = new_data
        if self.is_empty: # Adding first data
            self._add_first_data(x_t,z_t)
        else: # Updating the data
            self._add_new_data(x_t,z_t)  
    
    def update_batch(self,new_data_batch):
        """
            x_t = [x_t1,x_t2,...,x_tn] list of inputs
            z_t = [x_t1,x_t2,...,x_tn] list of outputs
        """
        #TODO : Assertions
        if self.is_empty:
            self.change_data(new_data_batch)
        else:
            x_t,z_t = new_data_batch
            num_new = len(x_t)
            self.numdata = self.numdata + num_new
            V = np.vstack([utils.relation_array(self.cov,x_ti,self.xdata) 
                           for x_ti in x_t]).transpose() #K(Xnew,Xold)
            C = utils.binary_function_matrix(self.cov,x_t) #K(Xnew,Xnew)
            if self.noisekernel.is_diagonal: #K(Xnew,Xnew) + sigma^2*I
                I = np.diag([self.noisecov(xx,xx) for xx in x_t])
                C = C + I
            else:
                raise NotImplementedError
            self.K = utilsla.expand_symmetric_with_matrix(self.K,V,C) #K
            self.U = utilsla.expand_cholesky_with_matrix(self.U,V,C) #cholesky factor
            self.xdata += x_t
            self.zdata = np.hstack([self.zdata,np.array(z_t)])
            self._update_likelihood()
    
    def optimize(self,adjustables,positives,trans = None,
                      verbose=0,rethypers = False,
                      tol=1e-4,method = "L-BFGS-B"):
        """
            Choose new parameters for the GP based on 
            MLE estimation.
            input:
                adjustables : [bool] list of parameters to change
                positives : [bool] list of positive parameters
                trans : list of transformation to 
                        be applied on ith parameters on optimization. 
                        options: None,"log","sqrt"
                        default: [None]
                rethypers: whether to return hyperparameters
                tol : solver tolerance.
                method : "L-BFGS-B","TNC" or "SLSQP"
            returns:
                GPObject with new parameters. If rethyper also hyperparameters
        """
        #TODO : For now not adjusting mean.
        #FIXME : This is turning quickly into spaghetti code
        #TODO : GET choose_samples functions and use here also
        if trans == None:
            trans = [None]*len(positives)
        def f_and_df(phi):
            if verbose >= 2: print(phi)
            # Returns both log likelihood and gradient 
            # of the gp with adjustables parameters changed
            phicopy = self.phi.copy()
            phicopy[adjustables] = phi
            for i,_ in enumerate(phicopy): #transformations
                if trans[i] == "log":
                    phicopy[i] = np.exp(phicopy[i])
                if trans[i] == "sqrt":
                    phicopy[i] = np.square(phicopy[i])
            gpcopy = GPObject(self.kernel,self.noisekernel,
                              phicopy,[self.xdata,self.zdata])
            f = -gpcopy.loglikelihood
            inds = [i for i in range(len(adjustables)) if adjustables[i]]
            df = -gpcopy._dbatch_loglikelihood(inds)
            for i,_ in enumerate(df): #transformations
                if trans[i] == "log":
                    df[i] = phicopy[i]*df[i]
                if trans[i] == "sqrt":
                    df[i] = 2*phicopy[i]*df[i] #TODO : Is this the correct way?
            return f,df
        newphiontrans = self.phi.copy()
        for i,_ in enumerate(newphiontrans): #transformations
            if trans[i] == "log":
                newphiontrans[i] = np.log(newphiontrans[i])
            if trans[i] == "sqrt":
                newphiontrans[i] = np.sqrt(newphiontrans[i])
        phiinit = newphiontrans[adjustables]
        #Setting bounds
        lb = []
        ub = []
        for i,adjustable in enumerate(adjustables):
            if adjustable:
                if trans[i] == "log":
                    lb.append(-25);ub.append(10)
                elif trans[i] == "sqrt":
                    lb.append(-np.inf);ub.append(np.inf)
                elif positives[i]:
                    lb.append(1e-20);ub.append(np.inf)
                else:
                    lb.append(-np.inf);ub.append(np.inf)
                ub.append(np.inf)
        bounds = spopt.Bounds(lb,ub)
        #Optimization
        opt = spopt.minimize(f_and_df,x0 = phiinit,jac = True,bounds=bounds,
                             tol = tol)
        if verbose >= 1 : print(opt)
        phiopt = opt.x
        newphi = self.phi.copy()
        newphi[adjustables] = phiopt
        for i,_ in enumerate(newphi): #transformations
            if trans[i] == "log":
                newphi[i] = np.exp(newphi[i])
            if trans[i] == "sqrt":
                newphi[i] = np.square(newphi[i])
        print(newphi)
        newgp = GPObject(self.kernel,self.noisekernel,
                            newphi,[self.xdata,self.zdata])
        if not rethypers:
            return newgp
        else:
            return newgp,newphi
    
    def _add_first_data(self,x_t,z_t):
        self.xdata = [x_t]
        self.zdata = np.array([z_t])
        self.numdata = 1
        self.K = np.array([[self.cov(x_t,x_t)]]) #K(X,X)
        self.K = self.K + self.noise_cov(x_t,x_t) #K(X,X) + sigma^2*I
        self.U = np.sqrt(self.K) #cholesky factor
        self.loglikelihood = -0.5*((z_t**2)/self.K[0,0] + \
                                   np.log(self.K[0,0]) + \
                                   self.numdata*np.log(2*np.pi))
        self.likelihood = np.exp(self.loglikelihood)
        self.is_empty = False
    
    def _add_new_data(self,x_t,z_t):
        self.numdata = self.numdata + 1
        v = utils.relation_array(self.cov,x_t,self.xdata) #k(xnew,Xold)
        c = self.cov(x_t,x_t) #k(xnew,xnew)
        c = c + self.noisecov(x_t,x_t) #k(xnew,xnew) + sigma^2
        self.K = utilsla.expand_symmetric(self.K,v,c) #K
        self.U = utilsla.expand_cholesky(self.U,v,c) #cholesky factor
        self.xdata.append(x_t)
        self.zdata = np.append(self.zdata,z_t)
        self._update_likelihood()
        
    def _initialize_kernels(self,kernel,noisekernel,phi):
        self.nhkern = kernel.nhyper #Number of kernel hyperparams
        self.nhnoise = noisekernel.nhyper #Number of noise kernel hyperparams
        assert len(phi) == self.nhkern + self.nhnoise #Check correct nhyper
        self.phi = phi #Hyperparams
        kernelphi,noisephi = phi[:self.nhkern],phi[self.nhkern:]
        self.kernel = copy.deepcopy(kernel)
        self.noisekernel = copy.deepcopy(noisekernel)
        if hasattr(kernel,'initialized') and kernel.initialized:
            self.kernel.reset()
        if hasattr(noisekernel,'initialized') and noisekernel.initialized:
            self.noisekernel.reset()
        self.kernel.initialize(kernelphi)
        self.noisekernel.initialize(noisephi)
        self.cov = self.kernel.f
        self.noisecov = self.noisekernel.f
        self.m = 0
    
    def _initialize_empty(self):
        self.xdata = None
        self.zdata = None
        self.numdata = 0
        self.K = None
        self.U = None
        self.likelihood = None
        self.loglikelihood = None
        self.is_empty = True
    
    def _update_likelihood(self):
        """
            Calculate likelihood
        """
        # TODO : add efficient update of r
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        l1 = np.dot(r,r)
        l2 = 2*np.sum(np.log(np.diag(self.U)))
        l3 = self.numdata*np.log(2*np.pi)
        self.loglikelihood = -0.5*(l1 + l2 + l3)

    def _dbatch_loglikelihood(self,inds):
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        alpha = utilsla.invumatmul(self.U,r,trans='N')
        M = np.outer(alpha,alpha)
        invK = utilsla.inverse_cholesky_upper(self.U)
        grads = []
        for i in inds:
            if i < self.nhkern: #Case kernel
                df = functools.partial(self.kernel.df,i=i)
            elif i >= self.nhkern: #Case noise kernel
                df = functools.partial(self.noisekernel.df,i=i - self.nhkern)
            dK = utils.binary_function_matrix(df,self.xdata)
            g = 0.5*((M-invK)*(dK.transpose())).sum()
            grads.append(g)
        return np.array(grads)
    
    def _d2batch_loglikelihood(self,inds):
        # See Gaussian Process for Regression and optimization
        # TODO : Better implementation
        invK = utilsla.inverse_cholesky_upper(self.U)
        c = np.matmul(invK,self.zdata - self.m) #invK*y
        grads2 = []
        for i in inds:
            if i < self.nhkern: #Case kernel
                df = functools.partial(self.kernel.df,i=i)
                d2f = functools.partial(self.kernel.d2f,i=i)
            elif i >= self.nhkern: #Case noise kernel
                df = functools.partial(self.noisekernel.df,i=i - self.nhkern)
                d2f = functools.partial(self.noisekernel.d2f,
                                       i=i - self.nhkern)
            dK = utils.binary_function_matrix(df,self.xdata)
            d2K = utils.binary_function_matrix(d2f,self.xdata)
            M = np.matmul(np.matmul(dK,invK),dK) #dK*invK*dK
            #First term. 1/2*tr(invK*(dK*invK*dK - d2K))
            term1 = 0.5*((M - d2K)*(invK.transpose())).sum()
            #Second term. 1/2*(c)'*(d2K - 2*dK*invK*dK)*c
            term2 = 0.5*utilsla.bilinear_form(c,d2K - 2*M,c)
            g2 = term1 + term2
            grads2.append(g2)
        return np.array(grads2)
            
    # Legacy functions
    def _dmean_loglikelihood(self):
        raise NotImplementedError # TODO : implement
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        alpha = utilsla.invumatmul(self.U,r,trans='N')
        g = np.sum(alpha)
        return g
    
    def _dnoisekernel_loglikelihood(self,i):
        #TODO : simplify
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        alpha = utilsla.invumatmul(self.U,r,trans='N')
        M = np.outer(alpha,alpha)
        invK = utilsla.inverse_cholesky_upper(self.U)
        df = functools.partial(self.noisekernel.df,i=i)
        dK = utils.binary_function_matrix(df,self.xdata)
        #1/2*tr((alpha*alpha^T - K^(-1))*dK)
        g = 0.5*((M-invK)*(dK.transpose())).sum()
        return g
    
    def _dkernel_loglikelihood(self,i):
        r = utilsla.invumatmul(self.U,self.zdata - self.m,trans='T')
        alpha = utilsla.invumatmul(self.U,r,trans='N')
        M = np.outer(alpha,alpha)
        invK = utilsla.inverse_cholesky_upper(self.U)
        df = functools.partial(self.kernel.df,i=i)
        dK = utils.binary_function_matrix(df,self.xdata)
        #1/2*tr((alpha*alpha^T - K^(-1))*dK)
        g = 0.5*((M-invK)*(dK.transpose())).sum()
        return g
    
    def _d2kernel_loglikelihood(self,i):
        raise NotImplementedError

        