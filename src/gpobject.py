# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import torch

from . import utils
from . import utilsla
from . import utilstorch
from . import gpoptimizer

LOG2PI = float(np.log(2*np.pi))


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
    
    def predict(self,x,**kwargs):
        getvar = kwargs.get("getvar",True)
        return_as_numpy = kwargs.get("return_as_numpy",True)
        retdiag = kwargs.get("retdiag",True)
        if np.shape(x) == ():
            return self.predict_single(x,getvar=getvar,
                                    return_as_numpy = return_as_numpy)
        else:
            if len(np.shape(x)) == 1:
                x = np.reshape(x,(-1,1))
            return self.predict_batch(x,getvar=getvar,
                                      return_as_numpy=return_as_numpy,
                                      retdiag=retdiag)

    def predict_single(self,x,getvar = True,return_as_numpy = True):
        """
            Calculate mean(x),var(x)
        """
        # TODO : add efficient update of r
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
            
    def optimize(self,option="B",**kwargs):
        """
            Choose new parameters for the GP based on 
            MLE estimation.
            input:
                option : "B","U", "B" stand for bounded, "U" for unbounded
                positives : [bool] list of positive parameters.
                            Required for unbounded optimization
                warpings: dictionary of warpings to apply to to 
                          index, of the form {i:option}, 
                          where option can be "sqrt","log"
                bounds : [(lb,ub),] list of lower and upper bounds. 
                         Used for bounded optimization
                num_starts : Number of times to run L-BFGS. 
                             Only available for bounded optimization.
                verbose : level of verbosity. Default : 1
            returns:
                GPObject with new parameters.
        """
        hparams_new =  gpoptimizer.optimize(self.kernel,self.noisekernel,
                         self.hparams,(self.xdata,self.ydata),
                         option,**kwargs)
        gpnew = GPObject(self.kernel,self.noisekernel,hparams_new,
                                  (self.xdata,self.ydata))
        return gpnew
        
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
