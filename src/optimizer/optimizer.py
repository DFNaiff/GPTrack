# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import torch

from . import lbfgs
from . import cg
from .. import utils
from .. import utilstorch


LOG2PI = float(np.log(2*np.pi))


def optimize(kernel,noisekernel,hparams,
              data,option,**kwargs):
    """
        Choose new parameters for the GP based on 
        MLE estimation.
        input:
            kernel : kernel of the GP
            noisekernel : kernel of the GP noise
            hparams : list of positive hyperparameters
            data : (xdata,ydata) tuple
            positives_default : how to deal with positive parameters.
                                if "bound", set [lb,infty] bound 
                                for each variable if bound not specified.
                                Then function takes a new argument 
                                plb, default : 1e-10
                                if "warp", set warping to positive parameters
                                Then takes a new argument, wlb, default: "log"
                                if None, the user has to deal with positives 
                                himself
            warpings : dict of warpings to be done to parameters. 
                      Options: "sqrt","log".
            bounds : dict of (lb,ub) lower and upper bounds. 
                     Used for bounded optimization
            frozen : list of parameters to be frozen, indexed by 
                     parameter number
            num_starts : Number of times to run L-BFGS. 
                         Only available for bounded optimization.
            verbose : level of verbosity. Default : 1
        returns:
            GPObject with new parameters.
    """
    bounds = kwargs.get("bounds")
    _param_opt = functools.partial(_optimize_single_start_b,
                    kernel = kernel,noisekernel = noisekernel,data=data)
    nll,hparams_new,bic = _param_opt(hparams = hparams,**kwargs)
    verbose = kwargs.get("verbose")
    if verbose >= 1:
        print(1,nll)
    if option == "B":
        num_starts = kwargs.get("num_starts",1)
        if num_starts > 1: raise NotImplementedError
        for i in range(1,num_starts):
            hparams_test = _sample_params(hparams,bounds)
            nll_test,hparams_test,bic_test = _param_opt(hparams = hparams_test,
                                                        **kwargs)
            if nll_test < nll:
                hparams_new = hparams_test
                nll = nll_test
                bic = bic_test
            if verbose >= 1:
                print(i+1,nll_test)
    return hparams_new,bic


def _optimize_single_start_b(kernel,noisekernel,hparams,
                           data,**kwargs):
    #The function that will be optimized
    def _f(hparams): #Function to be optimized
        hparams_feed = [None]*len(hparams)
        for i,_ in enumerate(hparams):
            if i in warp_dict: #Take off warping
                hparams_feed[i] = _inv_warp_transform(warp_dict[i])(hparams[i])
            elif positive_list[i] and positives_default == "warp":
                hparams_feed[i] = _inv_warp_transform(pwrp)(hparams[i])
            else:
                hparams_feed[i] = hparams[i].clone()
        if to_optimize == "likelihood":
            gpnew = NLLGPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = -gpnew.loglikelihood
        elif to_optimize == "loo_error":
            gpnew = _LOOGPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = gpnew._mse_loo_error()
        elif to_optimize == "crossvalidation":
            gpnew = _LOOGPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = gpnew._mse_cv_error(cvbatch)
        return result
    #Load necessary parameters
    verbose = kwargs.get("verbose")
    max_iter = kwargs.get("max_iter",100)
    opt_choice = kwargs.get("opt_choice","lbfgs")
    beta_update_fn = kwargs.get("beta_update_fn","PR")
    line_search_fn = kwargs.get("line_search_fn","goldstein")
    frozen = kwargs.get("frozen",list())
    to_optimize = kwargs.get("to_optimize","likelihood")
    positives_default = kwargs.get("positives_default","bound")
    frozen = kwargs.get("frozen",list())
    warp_dict = kwargs.get("warpings",dict())
    bounds_dict = kwargs.get("bounds",dict())
    pwrp = kwargs.get("pwrp","log")
    plb = kwargs.get("plb",1e-10)
    if to_optimize == "crossvalidation":
        cvbatch = kwargs.get("cvbatch",None)
        assert cvbatch != None
    
    #Prepare data and hyperparameters
    xdata,ydata = data
    positive_list = kernel.positives + noisekernel.positives
    #Prepare list of frozen indexes
    frozenlist = [False]*len(hparams)
    for ind in frozen:
        frozenlist[ind] = True
    if frozen == False:
        frozen = [False]*len(hparams)
    
    #Prepare bounds
    bounds = []
    for i,_ in enumerate(hparams):
        if i in bounds_dict:
            b = bounds_dict[i]
            if b[0] != None:
                b[0] = torch.tensor(b[0])
            if b[1] != None:
                b[1] = torch.tensor(b[1])
            bounds.append(b)
        elif positive_list[i] and positives_default == "bound":
            bounds.append([plb,None])
        else:
            bounds.append([None,None])
    
    #Adjust hparams so we can differentiate
    hparams_new = []
    for i,hparam in enumerate(hparams):
        hparam_new = hparam.clone()
        if i in warp_dict: #Warp
            hparam_new = _warp_transform(warp_dict[i])(hparam_new)
        elif positive_list[i] and positives_default == "warp":
            hparam_new = _warp_transform(pwrp)(hparam_new)
        hparam_new.requires_grad_()
        hparams_new.append(hparam_new)
    
    #Set the hparams to optimize (only those not frozen)
    notfrozen = utils.list_not(frozenlist)
    hparams_to_opt = utils.bool_slice(hparams_new,notfrozen)
    bounds_not_frozen = utils.bool_slice(bounds,notfrozen)
    
    #Optmizer
    if opt_choice == "lbfgs":
        optimizer = lbfgs.LBFGS(hparams_to_opt,max_iter=max_iter,
                                bounds=bounds_not_frozen,
                                line_search_fn = line_search_fn)
    elif opt_choice == "cg":
        optimizer = cg.CG(hparams_to_opt,max_iter=max_iter,
                          bounds=bounds_not_frozen,
                          beta_update_fn = beta_update_fn,
                          line_search_fn = line_search_fn)
    
    #Begin optimization
    optimizer.zero_grad()
    def closure(): #Closure function
        optimizer.zero_grad()
        if verbose >= 2:
            print([h.item() for h in hparams_new])
        nll = _f(hparams_new)
        nll.backward(retain_graph=True)
        if verbose >= 2:
            print(nll.item())
            print("-"*10)
        return nll
    loss = optimizer.step(closure)
    #Unwarp and remove grad from parameters
    for i,_ in enumerate(hparams_new):
        hparams_new[i].requires_grad = False
        if i in warp_dict: #Take off warping
            hparams_new[i] = _inv_warp_transform(warp_dict[i])(hparams_new[i])
        elif positive_list[i] and positives_default == "warp":
            hparams_new[i] = _inv_warp_transform(pwrp)(hparams_new[i])
    #Calculate Bayes information criterion
    bic = np.log(len(xdata))*len(hparams_to_opt) + 2*loss
    return loss,hparams_new,bic


#==============================================================================
# SIMPLIFIED GP OBJECTS FOR OPTIMIZATION
#==============================================================================
class _LOOGPObject(object):
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
        self.invK = torch.inverse(self.K)
        
    def _initialize_kernels(self,kernel,noisekernel,hparams):
        self.hparams = [None]*len(hparams)
        for i,_ in enumerate(hparams): #Convert to tensor
            if type(hparams[i]) == torch.Tensor:
                self.hparams[i] = hparams[i].clone()
            else:
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
        
    #LOO and cross validation functions
    def _loo_ll_error(self):
        """
            Calculates the leave one out error
        """
        #Invert matrix
        Ky = torch.matmul(self.K,self.ydata)
        error = 0
        for i in range(self.numdata):
            loo_single = self._loo_error_single(self.invK,Ky,i)
            error += loo_single
        return error
    
    def _loo_ll_error_single(self,invK,Ky,i):
        sigma2 = 1.0/invK[i,i]
        mu = self.ydata[i] - sigma2*Ky[i]
        term1 = -0.5*torch.log(sigma2)
        term2 = -0.5*(self.ydata[i] - mu)**2/sigma2
        term3 = -0.5*self.numdata/2*LOG2PI
        return term1 + term2 + term3
    
    def _mse_loo_error(self):
        """
            Calculate the leave one out error for mse
        """
        Ky = torch.matmul(self.K,self.ydata)
        error = torch.tensor(0.0)
        for i in range(self.numdata):
            loo_single = (Ky[i,0]/self.invK[i,i])**2
            error += loo_single/self.numdata
        return error
    
    def _mse_cv_error(self,cvbatch):
        Ky = torch.matmul(self.K,self.ydata)
        error = torch.tensor(0.0)
        for inds in cvbatch:
            cv_single = torch.sum(torch.gesv(Ky[inds],
                                             self.invK[np.ix_(inds,inds)])[0]**2)
            error += cv_single/self.numdata
        return error
        
#    #Input conversion
#    def _convert_input_single(self,x):
#        if type(x) != torch.Tensor:
#            return torch.tensor(x).float()
#        else:
#            return x.float()
#            
#    def _convert_input_batch(self,x):
#        if type(x) != torch.Tensor:
#            return torch.tensor(x).float()
#        else:
#            return x.float()
    
    #Print functions
    def showhparams(self):
        return [hp.item() for hp in self.hparams]


class NLLGPObject(object):
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
        self.is_empty = False
        
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
    
    #Print functions
    def showhparams(self):
        return [hp.item() for hp in self.hparams]

    def posteriori(self):
        return self.likelihood + self.kernel.ln_pdf() + \
               self.noisekernel.ln_pdf()
#==============================================================================
# AUXILIARY FUNCTIONS
#==============================================================================
def _sample_params(hparams,bounds):
    lb,ub = list(zip(*bounds))
    lb,ub = np.array(lb),np.array(ub)
    n = len(hparams)
    return list(np.random.random(n)*(ub - lb) + lb)

def _warp_transform(name):
    if name == "sqrt":
        return torch.sqrt
    if name == "log":
        return torch.log

def _inv_warp_transform(name):
    if name == "sqrt":
        return (lambda x : torch.pow(x,2))
    if name == "log":
        return torch.exp