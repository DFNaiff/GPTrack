# -*- coding: utf-8 -*-

# TODO : Implement d2f for all kernels
# TODO : Implement dfdx for all kernels
import numpy as np
import torch

from . import utils
from . import utilstorch


SQRT2 = float(np.sqrt(2))
SQRT3 = float(np.sqrt(3))
SQRT5 = float(np.sqrt(5))
PI = float(np.pi)


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
        self.dim = 0
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


class IsoMatern12(IsoRadKernel):
    """
        Isotropic Matern12 kernel. Number of hyperparams : 1
        {0:'l'}
    """
    def __init__(self,dim):
        super(IsoMatern12,self).__init__(dim)
        
    def f(self,x,y):
        r = torch.tensor(torch.sqrt(torch.sum((x - y)**2,1,keepdim=True)) / \
                self.hyperparams[0])
        return torch.exp(-r)


class IsoMatern32(IsoRadKernel):
    """
        Isotropic Matern52 kernel. Number of hyperparams : 1
        {0:'l'}
    """
    def __init__(self,dim):
        super(IsoMatern32,self).__init__(dim)
        
    def f(self,x,y):
        r = torch.tensor(torch.sqrt(torch.sum((x - y)**2,1,keepdim=True)) / \
                self.hyperparams[0])
        return (1 + SQRT3*r)*torch.exp(-SQRT3*r)
        

class IsoMatern52(IsoRadKernel):
    """
        Isotropic Matern52 kernel. Number of hyperparams : 1
        {0:'l'}
    """
    def __init__(self,dim):
        super(IsoMatern52,self).__init__(dim)
        
    def f(self,x,y):
        r = torch.tensor(torch.sqrt(torch.sum((x - y)**2,1,keepdim=True)) / \
                self.hyperparams[0])
        return (1 + SQRT5*r + 5.0/3*r**2)*torch.exp(-SQRT5*r)

#==============================================================================
# Periodic kernels
#==============================================================================
class PerKernel(Kernel):
    def __init__(self):
        self.dim = 1
        self.nhyper = 2
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False


class PerRBF(PerKernel):
    """
        Periodic RBF kernel. Number of hyperparams : 1
        {0:'T',1:'l'}
    """
    def __init__(self):
        super(PerRBF,self).__init__()
        
    def f(self,x,y):
        r2 = 4*torch.sin(PI*(x - y)/self.hyperparams[0])**2 / \
                   self.hyperparams[1]**2
#        print(r2)
        return torch.exp(-0.5*r2)


class PerMatern12(PerKernel):
    """
        Periodic Matern12 kernel. Number of hyperparams : 2
        {0:'T',1:'l'}
    """
    def __init__(self):
        super(PerMatern12,self).__init__()
        
    def f(self,x,y):
        #r2 = (cos(2*pi*x/T) - cos(2*pi*y/T))**2 + \
        #     (sin(2*pi*x/T) - sin(2*pi*y/T))**2
        r = 2*torch.abs(torch.sin(PI*(x - y)/self.hyperparams[0])) / \
                  self.hyperparams[1]
        return torch.exp(-r)


class PerMatern32(PerKernel):
    """
        Periodic Matern32 kernel. Number of hyperparams : 2
        {0:'T',1:'l'}
    """
    def __init__(self):
        super(PerMatern32,self).__init__()
        
    def f(self,x,y):
        r = 2*torch.abs(torch.sin(PI*(x - y)/self.hyperparams[0])) / \
                  self.hyperparams[1]
        return (1 + SQRT3*r)*torch.exp(-SQRT3*r)


class PerMatern52(PerKernel):
    """
        Periodic Matern52 kernel. Number of hyperparams : 2
        {0:'T',1:'l'}
    """
    def __init__(self,dim):
        super(PerMatern52,self).__init__()
    
    def f(self,x,y):
        r = 2*torch.abs(torch.sin(PI*(x - y)/self.hyperparams[0])) / \
                  self.hyperparams[1]
        return (1 + SQRT5*r + 5.0/3*r**2)*torch.exp(-SQRT5*r)


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
        self.dim = 1
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
        for i in range(1,self.nout):
            start_ind = utils.triangular(i-1)
            end_ind = utils.triangular(i)
            s = utilstorch.hypersphere_param(i+1,
                                             thetas[start_ind:end_ind])
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

class CholeskyCorr(Kernel):
    """
        Cholesky correlation kernel, with nout outputs.
    """
    def __init__(self,nout):
        raise NotImplementedError
        self.dim = 1
        self.nout = nout
        self.nhyper = nout*(nout+1)//2
        self.hyperparams = None
        self.W = None
        self.initialized = False

    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        S = torch.zeros((self.nout,self.nout))
        for i in range(0,self.nout):
            start_ind = utils.triangular(i)
            end_ind = utils.triangular(i+1)
            S[:i+1,i] = torch.tensor(hyperparams[start_ind:end_ind])
        W = torch.matmul(S.transpose(1,0),S) # S.T*S
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

class MONoiseKernel(Kernel):
    """
        Noise kernel for multiple outputs.
        
    """
    #TODO : Change
    def __init__(self,nout):
        self.nout = nout
        self.is_diagonal = True
        self.nhyper = nout
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self._htensor = torch.cat([torch.unsqueeze(hp,0) 
                                   for hp in self.hyperparams])
        self.noise_vars = hyperparams
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False
    
    def f(self,x,y):
        raise NotImplementedError
        return self.hyperparams[0]*\
               torch.prod(x == y,1,keepdim=True).float()
    
    def fdiag(self,x):
        #TODO : Strange mixed numpy/torch thing
        if type(x) == np.ndarray:
            return torch.tensor(self.hyperparams[x[:,0].astype(int)]*\
                                torch.ones(x.shape[0]))
        else:
            return self._htensor[x[:,0].long()]*torch.ones(x.size()[0])

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

#==============================================================================
# Special kernels
#==============================================================================
class ShiftedMO(Kernel):
    """
        Variation of a compound kernel made specifically 
        for shifted multiple outputs time series.
        Initializes with [k1.hyperparameters]
    """
    def __init__(self,k,nout,dim):
        self.k1 = SphericalCorr(nout)
        self.k2 = k
        self.dim = self.k1.dim + self.k2.dim
        self.nm = self.k1.dim
        self.nout = nout

    def initialize(self,hyperparams):
        assert len(hyperparams) == self.k1.nhyper + self.k2.nhyper + \
                                   self.nout - 1
        hyper1 = hyperparams[:self.k1.nhyper]
        hyper2 = hyperparams[self.k1.nhyper:self.k1.nhyper + self.k2.nhyper]
        hyper3 = hyperparams[self.k1.nhyper + self.k2.nhyper:]
        hyper3 = [torch.tensor(0.0)] + hyper3
        self._shiftparams = torch.cat([torch.unsqueeze(hp,0) 
                                       for hp in hyper3]).reshape(-1,1)
        self.k1.initialize(hyper1)
        self.k2.initialize(hyper2)
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.k1.reset()
        self.k2.reset()
        self.hyperparams = None
        self.initialized = False

    def f(self,x,y):
        tx = self._shiftparams[x[:,0].long()]*torch.ones((x.size()[0],1))
        ty = self._shiftparams[y[:,0].long()]*torch.ones((y.size()[0],1))
        return self.k1.f(x[:,:self.nm],y[:,:self.nm]) * \
               self.k2.f(x[:,self.nm:] - tx,y[:,self.nm:] - ty)
