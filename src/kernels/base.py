# -*- coding: utf-8 -*-
import torch

from . import prior

class Kernel(object):
    """
        In general : the kernel should be built 
                     from basic kernels, then LATER 
                     it should be initialized 
                     (the reason for this is in the 
                     implementation for GPTrack)
    """
    def __init__(self):
        pass
    
    def __add__(self,other):
        return Sum(self,other)
    
    def __mul__(self,other):
        return Prod(self,other)
    
    def initialize(self,hyperparams):
        """
            Must be overriden. hyperparams is 
            a list of tensor
        """
        raise NotImplementedError
    
    def set_priors(self,prior_dict={}):
        """
            prior dict must be in form {i:(prior_nm:str,prior_args:list)}
            if i not in prior dict, then prior is:
            normal(0,1), if not positive
            lognormal(0,1), if positive
            available priors:
                normal,lognormal,gamma,
                laplace,loglaplace,unif
        """
        self._prior_dict = dict()
        for i in range(self.nhyper):
            if i in prior_dict:
                self._prior_dict[i] = prior_dict[i]
            else:
                if self.positives[i]:
                    self._prior_dict[i] = ("lognormal",(0.0,1.0))
                else:
                    self._prior_dict[i] = ("normal",(0.0,1.0))
    
    def ln_pdf(self):
        if not hasattr(self,"_prior_dict"):
            raise ValueError("No priors")
        else:
            res = torch.tensor(0.0)
            for i,h in enumerate(self.hyperparams):
                name,args = self._prior_dict[i]
                res += prior.ln_pdf(h,name,args)
            return res

    def has_prior(self):
        return hasattr(self,"_prior_dict")
#==============================================================================
# Constant kernel
#==============================================================================
class Constant(Kernel):
    def __init__(self):
        """
            Constant kernel. Number of hyperparams : 1.
            {0:'h2'}
        """
        self.nhyper = 1
        self.dim = 0
        self.hyperparams = None
        self.initialized = False
        self.positives = [True]
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False

    def f(self,x,y):
        return self.hyperparams[0]
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            return 1.0

#==============================================================================
# Compositions
#==============================================================================
class CompoundKernel(Kernel):
    """
        Compound kernel
    """
    def __init__(self,k1,k2):
        self.k1 = k1
        self.k2 = k2
        self.nhyper = k1.nhyper + k2.nhyper
        self.hyperparams = None
        self.initialized = False
        self.positives = k1.positives + k2.positives
        
    def initialize(self,hyperparams):
        hyper1 = hyperparams[:self.k1.nhyper]
        hyper2 = hyperparams[self.k1.nhyper:]
        self.k1.initialize(hyper1)
        self.k2.initialize(hyper2)
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.k1.reset()
        self.k2.reset()
        self.hyperparams = None
        self.initialized = False
    
    def _get_hyperparams(self):
        return self.k1.hyperparams + self.k2.hyperparams

class Sum(CompoundKernel):
    """
        Sum kernel. k(x,y) = k1(x,y) + k2(x,y)
        Number of hyperparams : k1.hyper + k2.hyper
    """
    def __init__(self,k1,k2):
        self.dim = max(k1.dim,k2.dim)
        super(Sum,self).__init__(k1,k2)
        
    def f(self,x,y):
        return self.k1.f(x,y) + self.k2.f(x,y)


class Prod(CompoundKernel):
    """
        Hadamard product kernel. k(x,y) = k1(x,y)*k2(x,y)
        Number of hyperparams : k1.hyper + k2.hyper
    """
    def __init__(self,k1,k2):
        self.dim = max(k1.dim,k2.dim)
        super(Prod,self).__init__(k1,k2)

    def f(self,x,y):
        return self.k1.f(x,y)*self.k2.f(x,y)


class DirectSum(CompoundKernel):
    """
        Direct sum kernel. k([x1,y1],[x2,y2]) = k1(x1,y1) + k2(x2,y2)
    """
    def __init__(self,k1,k2):
        super(DirectSum,self).__init__(k1,k2)
        self.dim = k1.dim + k2.dim
        self.nm = k1.dim
        
    def f(self,x,y):
        return self.k1.f(x[:,:self.nm],y[:,:self.nm]) + \
               self.k2.f(x[:,self.nm:],y[:,self.nm:])


class TensorProd(CompoundKernel):
    """
        Direct product kernel. k([x1,y1],[x2,y2]) = k1(x1,y1)*k2(x2,y2)
    """
    def __init__(self,k1,k2):
        super(TensorProd,self).__init__(k1,k2)
        self.dim = k1.dim + k2.dim
        self.nm = k1.dim
        
    def f(self,x,y):
        return self.k1.f(x[:,:self.nm],y[:,:self.nm]) * \
               self.k2.f(x[:,self.nm:],y[:,self.nm:])


#    def __init__(self,k1,k2):
#        self.k1 = k1
#        self.k2 = k2
#        self.nhyper = k1.nhyper + k2.nhyper
#        self.hyperparams = None
#        self.initialized = False
#        self.positives = k1.positives + k2.positives
#        
#    def initialize(self,hyperparams):
#        hyper1 = hyperparams[:self.k1.nhyper]
#        hyper2 = hyperparams[self.k1.nhyper:]
#        self.k1.initialize(hyper1)
#        self.k2.initialize(hyper2)
#        self.hyperparams = hyperparams
#        self.initialized = True
#    
#    def reset(self):
#        self.k1.reset()
#        self.k2.reset()
#        self.hyperparams = None
#        self.initialized = False
