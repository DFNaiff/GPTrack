# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import torch

from . import lbfgs
from . import utils
from . import utilsla
from . import utilstorch


#TODO : You may be holding K on memory without needing
class GPObject(object):
    def __init__(self,kernel,noise_kernel,hparams,data,
                      **kwargs):
        """
            kernel : kernel of the GP
            noise_kernel : kernel of the GP noise
            hparams : list of hyperparameters (size: kernel.nparams +
                      noise_kernels.nparams)
            data : (xdata,ydata) tuple, where xdata is a 
                    (nsamples,nfeatures)-array and 
                    ydata is a (nsamples,1)-array
            gpmean : default mean of the GP
        """
        self.gpmean = kwargs.get("gpmean",0.0)
        self._initialize_kernels(kernel,noise_kernel,hparams)
        self.change_data(data)

    def predict(self,x,getvar = True,return_as_numpy = True):
        """
            Calculate mean(x),var(x)
        """
        # TODO : add efficient update of r
        # TODO : GIANT workaround here
        x = self._convert_input_single(x)
        kx = utilstorch.relation_array(self.kernel.f,x,self.xdata)
        s = torch.trtrs(kx,self.U,transpose=True)[0]
        mean = torch.matmul(s.transpose(1,0),self.z) + self.gpmean
        if not getvar:
            if return_as_numpy: return mean.numpy()[0][0]
            else: return mean[0][0]
        else:
            x = x.unsqueeze(0) # TODO : GIANT workaround here
            var = self.kernel.f(x,x) - torch.matmul(s.transpose(1,0),s)
            if return_as_numpy: return mean.numpy()[0][0],var.numpy()[0][0]
            else: return mean[0][0],var[0][0]
    
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
        mean = torch.matmul(s.transpose(1,0),self.z) + self.gpmean
        if not getvar:
            if return_as_numpy: return mean.numpy()
            else: return mean
        else:
            var = utilstorch.binary_function_matrix(self.kernel.f,x) - \
                  torch.matmul(s.transpose(1,0),s)
            if retdiag:
                var = torch.diag(var).reshape(-1,1)
            if return_as_numpy: return mean.numpy(),var.numpy()
            else: return mean,var
    
    def change_data(self,data):
        """
            Changes the data of the GP, replacing it with new_data
        """
        # TODO : assertions
        x,y = data
        self.xdata = torch.tensor(x).float()
        self.ydata = torch.tensor(y).float()
        self.numdata = self.xdata.shape[0]
        self.dimdata = self.xdata.shape[1]
        self.K = utilstorch.binary_function_matrix(self.kernel.f,
                                                   self.xdata)
        if self.noisekernel.is_diagonal: #K(X,X) + sigma2*I
            Idiag = self.noisekernel.fdiag(self.xdata)
            I = torch.diag(Idiag)
        else:
            I = self.noisekernel.fdiag(self.xdata)
        self.K = self.K + I
        self.U = torch.potrf(self.K)
        self._update_likelihood()
        self.is_empty = False
        
    def downdate(self,i=0):
        # TODO : Find a way to not pass through numpy
        self.K = torch.tensor(utilsla.contract(self.K.numpy(),i)).float()
        self.U = torch.tensor(utilsla.contract_cholesky(
                              self.U.numpy().astype(np.float),i)).float()
        # TODO : surely there's a function for dropping
        self.xdata = torch.cat([self.xdata[:i,:],self.xdata[i+1:,:]])
        self.ydata = torch.cat([self.ydata[:i],self.ydata[i+1:]])
        self.numdata = self.numdata - 1
        self._update_likelihood()
    
    def downdate_batch(self,drop_inds,is_sorted = False):
        """
            drop_inds = list of indexes to be dropped
        """
        # TODO : not clear whether there is a simple and more efficient way
        #        to downdate. But think of one
        if not is_sorted:
            drop_inds = sorted(drop_inds)
        for ind in drop_inds[::-1]:
            self.downdate(ind)

    def update_batch(self,new_data_batch):
        """
            new_data_batch : (xdata,ydata) tuple, where xdata is a 
                             (nsamples,nfeatures)-array and 
                             ydata is a (nsamples,1)-array
        """
        #TODO : Assertions
        x_new,y_new = new_data_batch
        x_new = torch.tensor(x_new).float()
        y_new = torch.tensor(y_new).float()
        num_new = x_new.shape[0]
        self.numdata = self.numdata + num_new
        V = utilstorch.binary_function_matrix_ret(self.kernel.f,self.xdata,x_new)
        C = utilstorch.binary_function_matrix(self.kernel.f,x_new)
        if self.noisekernel.is_diagonal: #K(X,X) + sigma2*I
            Idiag = self.noisekernel.fdiag(x_new)
            I = torch.diag(Idiag)
        else:
            I = self.noisekernel.fdiag(x_new)
        C = C + I

        self.K = torch.tensor(utilsla.expand_symmetric_with_matrix(self.K.numpy(),
                                                                   V.numpy(),
                                                                   C.numpy())).float() #K
        self.U = torch.tensor(utilsla.expand_cholesky_with_matrix(self.U.numpy(),
                                                                   V.numpy(),
                                                                   C.numpy())).float() #K
        self.xdata = torch.cat([self.xdata,x_new])
        self.ydata = torch.cat([self.ydata,y_new])
        self._update_likelihood()
    
    def optimize(self,option="B",**kwargs):
        """
            Choose new parameters for the GP based on 
            MLE estimation.
            input:
                option : "B","U", "B" stand for bounded, "U" for unbounded
                positives : [bool] list of positive parameters.
                            Required for unbounded optimization
                warpings: None,"sqrt","log". Only used for unbounded 
                          optimization. Default : "sqrt"
                bounds : [(lb,ub),] list of lower and upper bounds. 
                         Used for bounded optimization
                num_starts : Number of times to run L-BFGS. 
                             Only available for bounded optimization.
                verbose : level of verbosity. Default : 1
            returns:
                GPObject with new parameters.
        """
        return _optimize(self.kernel,self.noisekernel,
                         self.hparams,(self.xdata,self.ydata),
                         option,**kwargs)
        
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
def _optimize(kernel,noisekernel,hparams,
             data,option,**kwargs):
    """
        Choose new parameters for the GP based on 
        MLE estimation.
        input:
            kernel : kernel of the GP
            noisekernel : kernel of the GP noise
            hparams : list of positive hyperparameters
            data : (xdata,ydata) tuple
            option : "B","U", "B" stand for bounded, "U" for unbounded
            positives : [bool] list of positive parameters.
                        Required for unbounded optimization
            warpings: None,"sqrt","log". Only used for unbounded 
                      optimization. Default : "sqrt"
            bounds : [(lb,ub),] list of lower and upper bounds. 
                     Used for bounded optimization
            num_starts : Number of times to run L-BFGS. 
                         Only available for bounded optimization.
            verbose : level of verbosity. Default : 1
        returns:
            GPObject with new parameters.
    """
    #TODO : Check
    if option == "B":
        bounds = kwargs.get("bounds")
        if not bounds:
            raise TypeError("Bounds required for bounded optimizaton")
        _param_opt = functools.partial(_optimize_single_start_b,
                        kernel = kernel,noisekernel = noisekernel,data=data)
    elif option == "U":
        positives = kwargs.get("positives")
        if not positives:
            raise TypeError("Positives required for unbounded optimization")
        _param_opt = functools.partial(_optimize_single_start_u,
                                       kernel=kernel,noisekernel=noisekernel,
                                       data=data)
    nll,hparams_new = _param_opt(hparams = hparams,**kwargs)
    verbose = kwargs.get("verbose")
    if verbose >= 1:
        print(1,nll)
    if option == "B":
        num_starts = kwargs.get("num_starts",1)
        for i in range(1,num_starts):
            hparams_test = _sample_params(hparams,bounds)
            nll_test,hparams_test = _param_opt(hparams = hparams_test,
                                               **kwargs)
            if nll_test < nll:
                hparams_new = hparams_test
            if verbose >= 1:
                print(i+1,nll_test)
    gpnew = GPObject(kernel,noisekernel,hparams_new,
                              data)
    return gpnew


def _optimize_single_start_b(kernel,noisekernel,hparams,
                           data,**kwargs):
    #LBFG-S, no warping, Chang Yong-Oh implementation, bounded
    bounds = kwargs.get("bounds")
    verbose = kwargs.get("verbose")
    max_iter = kwargs.get("max_iter",100)
    line_search_fn = kwargs.get("line_search_fn","goldstein")
    xdata,ydata = data
    for i,_ in enumerate(hparams): #Convert to tensor
        hparams[i] = torch.tensor(hparams[i])
    for i,_ in enumerate(bounds):
        bounds[i] = (torch.tensor(bounds[i][0]),
                     torch.tensor(bounds[i][1]))
    def _negative_log_likelihood(hparams):
        hparams_feed = [None]*len(hparams)
        for i,_ in enumerate(hparams):
            hparams_feed[i] = hparams[i].clone()
        if verbose >= 2:
            print([h.item() for h in hparams_feed])
        try:
            gpnew = GPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = -gpnew.loglikelihood
        except RuntimeError:
            result = 1e12 + sum(hparams_feed)
        if verbose >= 2:
            print(result.item())
            print("-"*10)
        return result
    
    #Adjust hparams so we can differentiate
    hparams_new = []
    for i,hparam in enumerate(hparams):
        #TODO : Put adjustable here
        hparam_new = hparam.clone()
        hparam_new.requires_grad_()
        hparams_new.append(hparam_new)
    #Optmizer
    optimizer = lbfgs.LBFGS(hparams_new,max_iter=max_iter,bounds=bounds,
                            line_search_fn = line_search_fn)
    optimizer.zero_grad()
    def closure():
        optimizer.zero_grad()
        nll = _negative_log_likelihood(hparams_new)
        nll.backward(retain_graph=True)
        return nll
    nll = optimizer.step(closure)
    #Create new gp
    for i,_ in enumerate(hparams_new):
        hparams_new[i].requires_grad = False
    return nll.item(),hparams_new
    
def _optimize_single_start_u(kernel,noisekernel,hparams,
                           data,**kwargs):
    raise NotImplementedError

#==============================================================================
# AUXILIARY FUNCTIONS
#==============================================================================
def _sample_params(hparams,bounds):
    lb,ub = list(zip(*bounds))
    lb,ub = np.array(lb),np.array(ub)
    n = len(hparams)
    return list(np.random.random(n)*(ub - lb) + lb)