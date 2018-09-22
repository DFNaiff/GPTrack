# -*- coding: utf-8 -*-

# TODO : Implement d2f for all kernels
# TODO : Implement dfdx for all kernels
import numpy as np
import torch

from . import utils
from . import utilstorch

SQRT3 = np.sqrt(3)
SQRT5 = np.sqrt(5)


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
        
#==============================================================================
# Base kernels
#==============================================================================
class Constant(Kernel):
    def __init__(self):
        """
            Constant kernel. Number of hyperparams : 1.
            {0:'h2'}
        """
        self.nhyper = 1
        self.hyperparams = None
        self.initialized = False
    
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
# Radial kernels
#==============================================================================
class IsoRadKernel(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.nhyper = 1
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False


class IsoRBF(IsoRadKernel):
    """
        Isotropic RBF kernel. Number of hyperparams : 1
        {0:'l'}
    """
    def __init__(self,dim):
        super(IsoRBF,self).__init__(dim)
        
    def f(self,x,y):
        r2 = torch.tensor(torch.sum((x - y)**2,1,keepdim=True) / \
                (self.hyperparams[0]**2))
        return torch.exp(-0.5*r2)
#==============================================================================
# Multiple output kernels
#==============================================================================
class SphericalCorr(Kernel):
    """
        Spherical correlation kernel, with nout outputs. See 
        "Gaussian Process for Prediction" technical report, 
        by Michael Osborne. In initialization, 
        first nout parameters are the length scales, 
        where the others nout*(nout-1)/2 are the 
        correlation angles
    """
    def __init__(self,nout):
        self.nout = nout
        self.nhyper = nout + nout*(nout-1)//2
        self.hyperparams = None
        self.W = None
        self.initialized = False

    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        ls = hyperparams[:self.nout]
        ls = torch.cat([torch.unsqueeze(l,0) for l in ls])
        thetas = hyperparams[self.nout:]
        #Set correlation matrix
        S = torch.zeros((self.nout,self.nout))
        S[0,0] = 1.0
        th_temp = thetas.copy()
        for i in range(1,self.nout):
            th_now,th_temp = th_temp[:i],th_temp[i:]
            s = utilstorch.hypersphere_param(i+1,th_now)
            S[:i+1,i] = s
        W = torch.matmul(S.transpose(1,0),S) # S.T*S
        W = W*torch.ger(ls,ls) # diag(l)*S.T*S*diag(l)
        self.W = W
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.W = None
        self.initialized = False

    def f(self,x,y):
        return self.W[x.long(),y.long()] 


#==============================================================================
# Noise kernels
#==============================================================================
class IIDNoiseKernel(Kernel):
    """
        Regular noise kernel. Number of hyperparams : 1
        {0:'noise_var'}
    """
    def __init__(self):
        self.nhyper = 1
        self.is_diagonal = True
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False
    
    def f(self,x,y):
        return self.hyperparams[0]*\
               torch.prod(x == y,1,keepdim=True).float()
    
    def fdiag(self,x):
        #TODO : Strange mixed numpy/torch thing
        if type(x) == np.ndarray:
            return torch.tensor(self.hyperparams[0]*\
                                torch.ones(x.shape[0]))
        else:
            return self.hyperparams[0]*torch.ones(x.size()[0])
#==============================================================================
# Compound kernels
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
        super(Sum,self).__init__(k1,k2)
        
    def f(self,x,y):
        return self.k1.f(x,y) + self.k2.f(x,y)


class Prod(CompoundKernel):
    """
        Hadamard product kernel. k(x,y) = k1(x,y)*k2(x,y)
        Number of hyperparams : k1.hyper + k2.hyper
    """
    def __init__(self,k1,k2):
        super(Prod,self).__init__(k1,k2)

    def f(self,x,y):
        return self.k1.f(x,y)*self.k2.f(x,y)


class DirectSum(CompoundKernel):
    """
        Direct sum kernel. k([x1,y1],[x2,y2]) = k1(x1,y1) + k2(x2,y2)
    """
    def __init__(self,k1,k2):
        super(DirectSum,self).__init__(k1,k2)
        self.nm = k1.nhyper
        
    def f(self,x,y):
        return self.k1.f(x[:,:self.nm],y[:,:self.nm]) + \
               self.k2.f(x[:,:self.nm],y[:,:self.nm])


class TensorProd(CompoundKernel):
    """
        Direct product kernel. k([x1,y1],[x2,y2]) = k1(x1,y1)*k2(x2,y2)
    """
    def __init__(self,k1,k2):
        super(TensorProd,self).__init__(k1,k2)
        self.nm = k1.nhyper
        
    def f(self,x,y):
        return self.k1.f(x[:,:self.nm],y[:,:self.nm]) * \
               self.k2.f(x[:,:self.nm],y[:,:self.nm])