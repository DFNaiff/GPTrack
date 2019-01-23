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
    #Base functions
    def __init__(self,kernel,noise_kernel,hparams,data=None,
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
        if data:
            self.change_data(data)
        else:
            self.empty = True

    def predict(self,x,**kwargs):
        if self.empty: raise ValueError("GP has no data")
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
        if self.empty: raise ValueError("GP has no data")
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
        if self.empty: raise ValueError("GP has no data")
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
        self.xdata = utilstorch.convert_to_tensor_float(x)
        self.ydata = utilstorch.convert_to_tensor_float(y)
        self.numdata = self.xdata.shape[0]
        self.dimdata = self.xdata.shape[1]
        #TIP : Back to self.K if needed
        K = utilstorch.binary_function_matrix(self.kernel.f,
                                                   self.xdata)
        if self.noisekernel.is_diagonal: #K(X,X) + sigma2*I
            Idiag = self.noisekernel.fdiag(self.xdata)
            I = torch.diag(Idiag)
        else:
            I = self.noisekernel.fdiag(self.xdata)
        K = K + I
        self.U = torch.cholesky(K,upper=True)
        self._update_likelihood()
        self.empty = False
            
    def optimize(self,option="B",return_bic=False,**kwargs):
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
                Bayes Information Criterion calculated
        """
        if self.empty: raise ValueError("GP has no data")
        hparams_new,bic =  gpoptimizer.optimize(self.kernel,self.noisekernel,
                         self.hparams,(self.xdata,self.ydata),
                         option,**kwargs)
        gpnew = GPObject(self.kernel,self.noisekernel,hparams_new,
                                  (self.xdata,self.ydata))
        if return_bic:
            return gpnew,bic
        else:
            return gpnew

    #Updating functions    
    def update_batch(self,new_data_batch):
        """
            new data : (xdata,ydata) tuple, where xdata is a 
                    (nsamples,nfeatures)-array and 
                    ydata is a (nsamples,1)-array
        """
        #TODO : Assertions
        raise NotImplementedError
        if self.empty:
            self.change_data(new_data_batch)
        else:
            xnew,ynew = new_data_batch
            xnew = utilstorch.convert_to_tensor_float(xnew)
            ynew = utilstorch.convert_to_tensor_float(ynew)
            numnew = xnew.shape[0]
            self.xdata = torch.cat([self.xdata,xnew],0)
            self.ydata = torch.cat([self.ydata,ynew],0)
            self.numdata = self.xdata.shape[0]
            self.dimdata = self.xdata.shape[1]
            return numnew
#            num_new = len(x_t)
#            self.numdata = self.numdata + num_new
#            V = np.vstack([utils.relation_array(self.cov,x_ti,self.xdata) 
#                           for x_ti in x_t]).transpose() #K(Xnew,Xold)
#            C = utils.binary_function_matrix(self.cov,x_t) #K(Xnew,Xnew)
#            if self.noisekernel.is_diagonal: #K(Xnew,Xnew) + sigma^2*I
#                I = np.diag([self.noisecov(xx,xx) for xx in x_t])
#                C = C + I
#            else:
#                raise NotImplementedError
#            self.K = utilsla.expand_symmetric_with_matrix(self.K,V,C) #K
#            self.U = utilsla.expand_cholesky_with_matrix(self.U,V,C) #cholesky factor
#            self.xdata += x_t
#            self.ydata = np.hstack([self.ydata,np.array(z_t)])
#            self._update_likelihood()

    #Auxiliary functions
    def _initialize_kernels(self,kernel,noisekernel,hparams):
        self.hparams = [None]*len(hparams)
        for i,_ in enumerate(hparams): #Convert to tensor
            self.hparams[i] = utilstorch.convert_to_tensor_float(hparams[i])
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
        self.z = torch.trtrs(self.ydata.float(),
                             self.U,transpose=True)[0]
        term1 = 0.5*torch.sum(self.z**2)
        term2 = torch.sum(torch.log(torch.diag(self.U)))
        term3 = 0.5*self.numdata*np.log(2*np.pi)
        self.loglikelihood = -(term1 + term2 + term3)

    #Input conversion
    def _convert_input_single(self,x):
        return utilstorch.convert_to_tensor_float(x)
            
    def _convert_input_batch(self,x):
        return utilstorch.convert_to_tensor_float(x)

    #Print functions
    def showhparams(self):
        return [hp.item() for hp in self.hparams]
        
    #Other
    def logposteriori(self):
        return self.loglikelihood + self.kernel.ln_pdf() + \
               self.noisekernel.ln_pdf()