# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import torch

from . import lbfgs
from . import utils
from . import utilsla
from . import utilstorch


#TODO : Find a way to everything be a tensor. For now 
#       strange numpy/tensor mix (mainly because of the kernel matrix)

#TODO : Operate only on tensor in GPObject. Spit numpy if needed
class GPObject(object):
    def __init__(self,kernel,noise_kernel,hparams,data = None):
        self._initialize_kernels(kernel,noise_kernel,hparams)
        self.change_data(data)

    def predict(self,x,getvar = True,return_as_numpy = True):
        """
            Calculate mean(x),var(x)
        """
        # TODO : add efficient update of r
        x = self._convert_input_single(x)
        kx = utilstorch.relation_array(self.kernel.f,x,self.xdata)
        s = torch.trtrs(kx,self.U,transpose=True)[0]
        mean = torch.matmul(s.transpose(1,0),self.z)
        if not getvar:
            if return_as_numpy: return mean.numpy()
            else: return mean
        else:
            var = self.kernel.f(x,x) - torch.matmul(s.transpose(1,0),s)
            if return_as_numpy: return mean.numpy(),var.numpy()
            else: return mean,var
    
    def predict_batch(self,x,getvar = True,return_as_numpy = True,
                       retdiag = True):
        """
            Calculate mean(x),var(x)
        """
        # TODO : add efficient update of r
        x = self._convert_input_batch(x)
        kx = utilstorch.binary_function_matrix_ret(self.kernel.f,
                                                   self.xdata,x)
        s = torch.trtrs(kx,self.U,transpose=True)[0]
        mean = torch.matmul(s.transpose(1,0),self.z)
        if not getvar:
            if return_as_numpy: return mean.numpy()
            else: return mean
        else:
            var = utilstorch.binary_function_matrix(self.kernel.f,x) - \
                  torch.matmul(s.transpose(1,0),s)
            if retdiag:
                var = torch.diag(var)
            if return_as_numpy: return mean.numpy(),var.numpy()
            else: return mean,var
    
    def change_data(self,data):
        """
            Changes the data of the GP, replacing it with new_data
        """
        # TODO : assertions
        x,y = data
        self.xdata = torch.tensor(x)
        self.ydata = torch.tensor(y)
        self.numdata = x.shape[0]
        self.dimdata = x.shape[1]
        self.K = utilstorch.binary_function_matrix(self.kernel.f,
                                                   self.xdata)
        if self.noisekernel.is_diagonal: #K(X,X) + sigma2*I
            Idiag = self.noisekernel.fdiag(self.xdata)
            I = torch.diag(Idiag)
            self.K = self.K + I
        else:
            I = self.noisekernel.fdiag(self.xdata)
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
    
    def optimize(self,positives,adjustable=True,
                      num_starts = 1,verbose=False,
                      penalize_runtime_error = True,
                      option = "A",bounds = None):
        """
            Choose new parameters for the GP based on 
            MLE estimation.
            input:
                positives : [bool] list of positive parameters
                adjustables : [bool] list of parameters to change
                num_starts : Number of times to run L-BFGS
            returns:
                GPObject with new parameters. If rethyper also hyperparameters
        """
        return _optimize(self.kernel,self.noisekernel,
                                    self.hparams,(self.xdata,self.ydata),
                                    positives,adjustable,
                                    num_starts,verbose,
                                    penalize_runtime_error,
                                    option = option,bounds=bounds)

    def _add_new_data(self,x_t,z_t):
        raise NotImplementedError
        self.numdata = self.numdata + 1
        v = utils.relation_array(self.cov,x_t,self.xdata) #k(xnew,Xold)
        c = self.cov(x_t,x_t) #k(xnew,xnew)
        c = c + self.noisecov(x_t,x_t) #k(xnew,xnew) + sigma^2
        self.K = utilsla.expand_symmetric(self.K,v,c) #K
        self.U = utilsla.expand_cholesky(self.U,v,c) #cholesky factor
        self.xdata.append(x_t)
        self.ydata = np.append(self.ydata,z_t)
        self._update_likelihood()
        
    def _initialize_kernels(self,kernel,noisekernel,hparams):
        self.hparams = [None]*len(hparams)
        for i,_ in enumerate(hparams): #Convert to tensor
            self.hparams[i] = torch.tensor(hparams[i])
        self.nhkern = kernel.nhyper #Number of kernel hyperparams
        self.nhnoise = noisekernel.nhyper #Number of noise kernel hyperparams
        assert len(hparams) == self.nhkern + self.nhnoise #Check correct nhyper
        kernelparams = self.hparams[:self.nhkern]
        noiseparams = self.hparams[self.nhkern:]
        #Set kernels
        self.kernel = copy.deepcopy(kernel)
        self.noisekernel = copy.deepcopy(noisekernel)
        if hasattr(kernel,'initialized') and kernel.initialized:
            self.kernel.reset()
        if hasattr(noisekernel,'initialized') and noisekernel.initialized:
            self.noisekernel.reset()
        self.kernel.initialize(kernelparams)
        self.noisekernel.initialize(noiseparams)
    
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
    
    #Input conversion
    def _convert_input_single(self,x):
        return torch.tensor(x).float()

    def _convert_input_batch(self,x):
        return torch.tensor(x).float()
    
    #Print functions
    def showhparams(self):
        return [hp.item() for hp in self.hparams]


#==============================================================================
# Optimizer for GP
#==============================================================================
#TODO : Due to circular dependences, this isn't in another .py file.
#       But there should be a workaround
#TODO : DRY
def _optimize(kernel,noisekernel,hparams,
             data,positives,adjustable=True,
             num_starts = 1,verbose=False,
             penalize_runtime_error = True,
             option = "A",bounds=None):
    """
        Choose new parameters for the GP based on 
        MLE estimation.
        input:
            positives : [bool] list of positive parameters
            adjustables : [bool] list of parameters to change
            num_starts : Number of times to run L-BFGS
        returns:
            GPObject with new parameters. If rethyper also hyperparameters
    """
    #TODO : Check
    if option == "A":
        fopt = _optimize_single_start_a
    elif option == "B":
        fopt = _optimize_single_start_b
    elif option == "C":
        fopt = functools.partial(_optimize_single_start_c,
                                 bounds=bounds)
    _param_opt = functools.partial(fopt,
                    kernel = kernel,noisekernel = noisekernel,data=data,
                    positives = positives,adjustable = adjustable,
                    verbose = verbose, 
                    penalize_runtime_error = penalize_runtime_error)
    nll,hparams_new = _param_opt(hparams = hparams)
    for i in range(1,num_starts):
        hparams_test = _perturb(hparams,positives)
        nll_test,hparams_test = _param_opt(hparams = hparams_test)
        if nll_test < nll:
            hparams_new = hparams_test
        print(nll_test,hparams_new)
    gpnew = GPObject(kernel,noisekernel,hparams_new,
                              data)
    return gpnew


def _optimize_single_start_a(kernel,noisekernel,hparams,
                             data,positives,adjustable=True,
                             verbose=False,
                             penalize_runtime_error = True):
    #LBFG-S, with warping function sqrt    
    xdata,ydata = data
    for i,_ in enumerate(hparams): #Convert to tensor
        hparams[i] = torch.tensor(hparams[i])
    
    def _negative_log_likelihood(hparams,positives):
        hparams_feed = [None]*len(hparams)
        for i,_ in enumerate(hparams):
            if positives[i]:
                hparams_feed[i] = hparams[i]**2
            else:
                hparams_feed[i] = hparams[i].clone()
        if verbose:
            print([h.item() for h in hparams_feed])
        try:
            gpnew = GPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = -gpnew.loglikelihood
        except RuntimeError:
            result = 1e12 + sum(hparams_feed)
        if verbose:
            print(result.item())
            print("-"*10)
        return result
    
    if adjustable == True:
        adjustable = True*len(positives)
    #Adjust hparams so we can differentiate
    hparams_new = []
    for i,hparam in enumerate(hparams):
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
    nll = optimizer.step(closure)
    #Create new gp
    for i,_ in enumerate(hparams_new):
        hparams_new[i].requires_grad = False
        if positives[i]:
            hparams_new[i] = hparams_new[i]**2
    return nll.item(),hparams_new


def _optimize_single_start_b(kernel,noisekernel,hparams,
                             data,positives,adjustable=True,
                             verbose=False,
                             penalize_runtime_error = True):
    #LBFG-S, with warping function log    
    xdata,ydata = data
    for i,_ in enumerate(hparams): #Convert to tensor
        hparams[i] = torch.tensor(hparams[i])
    
    def _negative_log_likelihood(hparams,positives):
        hparams_feed = [None]*len(hparams)
        for i,_ in enumerate(hparams):
            if positives[i]:
                hparams_feed[i] = torch.exp(hparams[i])
            else:
                hparams_feed[i] = hparams[i].clone()
        if verbose:
            print([h.item() for h in hparams_feed])
        try:
            gpnew = GPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = -gpnew.loglikelihood
        except RuntimeError:
            result = 1e12 + sum(hparams_feed)
        if verbose:
            print(result.item())
            print("-"*10)
        return result
    
    if adjustable == True:
        adjustable = True*len(positives)
    #Adjust hparams so we can differentiate
    hparams_new = []
    for i,hparam in enumerate(hparams):
        #TODO : Put adjustable here
        hparam_new = hparam.clone()
        if positives[i]: #TODO : Check
            hparam_new = torch.log(hparam).clone()
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
    nll = optimizer.step(closure)
    #Create new gp
    for i,_ in enumerate(hparams_new):
        hparams_new[i].requires_grad = False
        if positives[i]:
            hparams_new[i] = hparams_new[i]**2
    return nll.item(),hparams_new


def _optimize_single_start_c(kernel,noisekernel,hparams,
                             data,positives,adjustable=True,
                             verbose=False,
                             penalize_runtime_error = True,
                             bounds = None):
    #LBFG-S, no warping, Chang Yong-Oh implementation, bounded
    xdata,ydata = data
    for i,_ in enumerate(hparams): #Convert to tensor
        hparams[i] = torch.tensor(hparams[i])
    for i,_ in enumerate(bounds):
        bounds[i] = (torch.tensor(bounds[i][0]),
                     torch.tensor(bounds[i][1]))
    def _negative_log_likelihood(hparams,positives):
        hparams_feed = [None]*len(hparams)
        for i,_ in enumerate(hparams):
            hparams_feed[i] = hparams[i].clone()
        if verbose:
            print([h.item() for h in hparams_feed])
        try:
            gpnew = GPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = -gpnew.loglikelihood
        except RuntimeError:
            result = 1e12 + sum(hparams_feed)
        if verbose:
            print(result.item())
            print("-"*10)
        return result
    
    if adjustable == True:
        adjustable = True*len(positives)
    #Adjust hparams so we can differentiate
    hparams_new = []
    for i,hparam in enumerate(hparams):
        #TODO : Put adjustable here
        hparam_new = hparam.clone()
        hparam_new.requires_grad_()
        hparams_new.append(hparam_new)
    #Optmizer
    optimizer = lbfgs.LBFGS(hparams_new,max_iter=100,bounds=bounds,
                            line_search_fn = "backtracking")
    optimizer.zero_grad()
    def closure():
        optimizer.zero_grad()
        nll = _negative_log_likelihood(hparams_new,positives)
        nll.backward(retain_graph=True)
        return nll
    nll = optimizer.step(closure)
    #Create new gp
    for i,_ in enumerate(hparams_new):
        hparams_new[i].requires_grad = False
    return nll.item(),hparams_new
    

#==============================================================================
# AUXILIARY FUNCTIONS
#==============================================================================
def _perturb(hparams,positives,noise = 0.1):
    hparams_perturb = [None]*len(hparams)
    for i,p in enumerate(hparams):
        if positives[i]:
            hparams_perturb[i] = (np.sqrt(p) + noise*np.random.randn())**2
        else:
            hparams_perturb[i] = p + noise*np.random.randn()
    return hparams_perturb