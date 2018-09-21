# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import scipy.optimize as spopt
import torch

from . import utils
from . import utilsla
from . import utilstorch


#TODO : Find a way to everything be a tensor. For now 
#       strange numpy/tensor mix (mainly because of the kernel matrix)

class GPObject(object):
    def __init__(self,kernel,noise_kernel,phi,data = None):
        self._initialize_kernels(kernel,noise_kernel,phi)
        self.change_data(data)
    
    def predict(self,x,getvar = True):
        """
            Calculate mean(x),var(x)
        """
        # TODO : add efficient update of r
        x_t = torch.tensor(x).float()
        kx = utilstorch.relation_array(self.kernel.f,x,self.xdata)
        s = torch.trtrs(kx,self.U,transpose=True)[0]
        mean = torch.matmul(s.transpose(1,0),self.z)
        if not getvar:
            return mean
        else:
            var = self.kernel.f(x_t,x_t) - torch.matmul(s.transpose(1,0),s)
            return mean,var
    
    def predict_batch(self,xs):
        raise NotImplementedError
        Kx = utils.binary_function_matrix_2(self.cov,xs,self.xdata)
        r = utilsla.invumatmul(self.U,self.ydata - self.m,trans='T')
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
        x,y = data
        self.xdata = x.copy()
        self.ydata = y.copy()
        self.numdata = x.shape[0]
        self.dimdata = x.shape[1]
        self.K = utilstorch.binary_function_matrix(self.kernel.f,
                                                   self.xdata)
        if self.noisekernel.is_diagonal: #K(X,X) + sigma2*I
            I = torch.diag(self.noisekernel.fdiag(self.xdata).flatten())
            self.K = self.K + I
        else:
            raise NotImplementedError
        self.U = torch.potrf(self.K)
        self._update_likelihood()
        self.is_empty = False
        
    def downdate(self,i=0):
        raise NotImplementedError
        # TODO : do not simply downdate likelihood
        self.K = utilsla.contract(self.K,i)
        self.U = utilsla.contract_cholesky(self.U,i)
        self.xdata = self.xdata[:i] + self.xdata[i+1:]
        # TODO : surely there's a function for dropping
        self.ydata = np.concatenate([self.ydata[:i],
                                     self.ydata[i+1:]])
        self._update_likelihood()
    
    def downdate_batch(self,drop_inds,is_sorted = True):
        raise NotImplementedError
        # TODO : not clear whether there is a simple and more efficient way
        #        to downdate. But think of one
        # Assumes drop_inds to be sorted
        if not is_sorted:
            drop_inds = sorted(drop_inds)
        for ind in drop_inds[::-1]:
            self.downdate(ind)
            
    def update(self,new_data):
        raise NotImplementedError
        """
            Updates data, covariance matrix and it's cholesky factor, 
            and likelihood
        """
        x_t,z_t = new_data
        self._add_new_data(x_t,z_t)  
    
    def update_batch(self,new_data_batch):
        raise NotImplementedError
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
            self.ydata = np.hstack([self.ydata,np.array(z_t)])
            self._update_likelihood()
    
    def optimize(self,positives,adjustable=True):
        """
            Choose new parameters for the GP based on 
            MLE estimation.
            input:
                positives : [bool] list of positive parameters
                adjustables : [bool] list of parameters to change
            returns:
                GPObject with new parameters. If rethyper also hyperparameters
        """
        #REMOVE THIS : JUST FOR DEBUGGING
        #TODO : Check
        def _negative_log_likelihood(hparams,positives):
            hparams_feed = [None]*len(hparams)
            for i,_ in enumerate(hparams):
                if positives[i]:
                    hparams_feed[i] = hparams[i]**2
                else:
                    hparams_feed[i] = hparams[i].clone()
            gpnew = GPObject(self.kernel,self.noisekernel,hparams_feed,
                             (self.xdata,self.ydata))
            print(-gpnew.loglikelihood.item())
#            print(-gpy_log_likelihood(self.xdata,self.ydata,
#                                     hparams[0],hparams[1],hparams[2]))
            print([p.item() for p in hparams_feed])
#            print('---')
            return -gpnew.loglikelihood
        
        if adjustable == True:
            adjustable = True*len(positives)
        #Adjust hparams so we can differentiate
        hparams_new = []
        for i,hparam in enumerate(self.phi):
            #TODO : Put adjustable here
            hparam_new = hparam.clone()
            if positives[i]: #TODO : Check
                hparam_new = torch.sqrt(hparam).clone()
            hparam_new.requires_grad_()
            hparams_new.append(hparam_new)
        #Optmizer
        optimizer = torch.optim.LBFGS(hparams_new,max_iter=100)
        optimizer.zero_grad()
        def closure():
            optimizer.zero_grad()
            nll = _negative_log_likelihood(hparams_new,positives)
            nll.backward(retain_graph=True)
            return nll
        optimizer.step(closure)
        #Create new gp
        for i,_ in enumerate(hparams_new):
            hparams_new[i].requires_grad = False
            if positives[i]:
                hparams_new[i] = hparams_new[i]**2        
        gpnew = GPObject(self.kernel,self.noisekernel,hparams_new,
                         (self.xdata,self.ydata))
        return gpnew


    def _add_new_data(self,x_t,z_t):
        self.numdata = self.numdata + 1
        v = utils.relation_array(self.cov,x_t,self.xdata) #k(xnew,Xold)
        c = self.cov(x_t,x_t) #k(xnew,xnew)
        c = c + self.noisecov(x_t,x_t) #k(xnew,xnew) + sigma^2
        self.K = utilsla.expand_symmetric(self.K,v,c) #K
        self.U = utilsla.expand_cholesky(self.U,v,c) #cholesky factor
        self.xdata.append(x_t)
        self.ydata = np.append(self.ydata,z_t)
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
    
    def _update_likelihood(self):
        """
            Calculate likelihood
        """
        self.z = torch.trtrs(torch.tensor(self.ydata).float(),
                             self.U,transpose=True)[0]
        term1 = 0.5*torch.sum(self.z**2)
        term2 = torch.sum(torch.log(torch.diag(self.U)))
        term3 = 0.5*self.numdata*np.log(2*np.pi)
        self.loglikelihood = -(term1 + term2 + term3)

    def _dbatch_loglikelihood(self,inds):
        r = utilsla.invumatmul(self.U,self.ydata - self.m,trans='T')
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
        c = np.matmul(invK,self.ydata - self.m) #invK*y
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
        r = utilsla.invumatmul(self.U,self.ydata - self.m,trans='T')
        alpha = utilsla.invumatmul(self.U,r,trans='N')
        g = np.sum(alpha)
        return g
    
    def _dnoisekernel_loglikelihood(self,i):
        #TODO : simplify
        r = utilsla.invumatmul(self.U,self.ydata - self.m,trans='T')
        alpha = utilsla.invumatmul(self.U,r,trans='N')
        M = np.outer(alpha,alpha)
        invK = utilsla.inverse_cholesky_upper(self.U)
        df = functools.partial(self.noisekernel.df,i=i)
        dK = utils.binary_function_matrix(df,self.xdata)
        #1/2*tr((alpha*alpha^T - K^(-1))*dK)
        g = 0.5*((M-invK)*(dK.transpose())).sum()
        return g
    
    def _dkernel_loglikelihood(self,i):
        r = utilsla.invumatmul(self.U,self.ydata - self.m,trans='T')
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

        