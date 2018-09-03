# -*- coding: utf-8 -*-

# TODO : Implement d2f for all kernels
# TODO : Implement dfdx for all kernels
import numpy as np

from . import utils


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

#==============================================================================
# Base kernels
#==============================================================================
class Constant(Kernel):
    def __init__(self):
        """
            Constant kernel. Number of hyperparams : 1
        """
        self.nhyper = 1
        self.hyperparams = None
        self.theta = None
        self.initialized = False
    
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.theta = hyperparams[0]
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.theta = None
        self.initialized = False

    def f(self,x,y):
        return self.theta
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            return 1.0
    
    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            return 0.0
    
    def dfdx(self,x,y):
        return 0.0
#==============================================================================
# Radial kernels
#==============================================================================
class IsoRadKernel(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.nhyper = 1
        self.l = None
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.l = hyperparams[0]
        self.initialized = True
    
    def reset(self):
        self.l = None
        self.hyperparams = None
        self.initialized = False


class IsoRBF(IsoRadKernel):
    """
        Isotropic RBF kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        super(IsoRBF,self).__init__(dim)
        
    def f(self,x,y):
        r2 = np.sum((x-y)**2)/(self.l**2)
        return np.exp(-0.5*r2)
        
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r2 = np.sum((x-y)**2)/(self.l**2)
            dr2dl = -2*r2/self.l
            return -0.5*np.exp(-0.5*r2)*dr2dl
    
    def d2f(self,x,y,i): #d2f/di2
        assert i >= 0 and i < self.nhyper
        if i == 0: #FIXME : Language not consistent
            d2 = np.sum((x-y)**2)
            return d2*(d2 - 3*self.l**2)*\
                   np.exp(-0.5*d2/(self.l**2))/(self.l**6)
        
    def dfdx(self,x,y):
        r2 = np.sum((x-y)**2)/(self.l**2)
        return -np.exp(-0.5*r2)*(x - y)/(self.l**2)


class IsoMatern12(IsoRadKernel):
    """
        Isotropic Matern12 kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        super(IsoMatern12,self).__init__(dim)
        
    def f(self,x,y):
        r = np.sqrt(np.sum((x-y)**2))/self.l
        return np.exp(-r)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r = np.sqrt(np.sum((x-y)**2))/self.l
            drdl = -r/self.l
            return -np.exp(-r)*drdl
    
    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r = np.sqrt(np.sum((x-y)**2))/self.l
            drdl = -r/self.l
            dr2dl = 2*r/(self.l**2)
            dfdr = -np.exp(-r)
            d2fdr = np.exp(-r)
            return d2fdr*(drdl)**2 + dfdr*dr2dl
            

class IsoMatern32(IsoRadKernel):
    """
        Isotropic Matern32 kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        super(IsoMatern32,self).__init__(dim)
        
    def f(self,x,y):
        r = np.sqrt(np.sum((x-y)**2))/self.l
        return (1 + SQRT3*r)*np.exp(-SQRT3*r)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r = np.sqrt(np.sum((x-y)**2))/self.l
            drdl = -r/self.l
            return -3*r*np.exp(-SQRT3*r)*drdl

    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r = np.sqrt(np.sum((x-y)**2))/self.l
            drdl = -r/self.l
            dr2dl = 2*r/(self.l**2)
            dfdr = -3*r*np.exp(-SQRT3*r)
            d2fdr = 3*(SQRT3*r - 1)*np.exp(-SQRT3*r)
            return d2fdr*(drdl)**2 + dfdr*dr2dl


class IsoMatern52(IsoRadKernel):
    """
        Isotropic Matern52 kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        super(IsoMatern52,self).__init__(dim)
    
    def f(self,x,y):
        r = np.sqrt(np.sum((x-y)**2))/self.l
        return (5.0*r**2/3 + SQRT5*r + 1)*np.exp(-SQRT5*r)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r = np.sqrt(np.sum((x-y)**2))/self.l
            drdl = -r/self.l
            return -5.0*r*(SQRT5*r + 1)*np.exp(-SQRT5*r)/3*drdl

    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r = np.sqrt(np.sum((x-y)**2))/self.l
            drdl = -r/self.l
            dr2dl = 2*r/(self.l**2)
            dfdr = -5.0*r*(SQRT5*r + 1)*np.exp(-SQRT5*r)/3
            d2fdr = 5.0*(5*r**2 - 1*SQRT5*r - 1)*np.exp(-SQRT5*r)/3
            return d2fdr*(drdl)**2 + dfdr*dr2dl


class AnisoRadKernel(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.nhyper = dim
        self.l = None
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.l = hyperparams
        self.initialized = True
    
    def reset(self):
        self.l = None
        self.hyperparams = None
        self.initialized = False


class AnisoRBF(AnisoRadKernel):
    """
        Isotropic RBF kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        super(AnisoRBF,self).__init__(dim)
        
    def f(self,x,y):
        r2 = np.sum((x-y)**2/(self.l**2))
        return np.exp(-0.5*r2)
        
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r2 = np.sum((x-y)**2)/(self.l**2)
            dr2dl = -2*(x[i] - y[i])/(self.l[i]**3)
            return -0.5*np.exp(-0.5*r2)*dr2dl
    
    def d2f(self,x,y,i): #d2f/di2
        raise NotImplementedError
        assert i >= 0 and i < self.nhyper
        if i == 0: #FIXME : Language not consistent
            d2 = np.sum((x-y)**2)
            return d2*(d2 - 3*self.l**2)*\
                   np.exp(-0.5*d2/(self.l**2))/(self.l**6)


#==============================================================================
# Periodic Kernels
#==============================================================================
class PerKernel(Kernel):
    def __init__(self):
        self.nhyper = 1
        self.T = None
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.T = hyperparams[0]
        self.initialized = True
    
    def reset(self):
        self.T = None
        self.hyperparams = None
        self.initialized = False


class PerRBF(PerKernel):
    """
        Periodic RBF kernel. Number of hyperparams : 1
    """
    def __init__(self):
        super(PerRBF,self).__init__()
        
    def f(self,x,y):
        #r2 = (cos(2*pi*x/T) - cos(2*pi*y/T))**2 + \
        #     (sin(2*pi*x/T) - sin(2*pi*y/T))**2
        r2 = -2*np.cos(2*np.pi*(x - y)/self.T) + 2
        return np.exp(-0.5*r2)
        
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            r2 = -2*np.cos(2*np.pi*(x - y)/self.T) + 2
            dr2dT = -4*np.pi*(x - y)*\
                     np.sin(2*np.pi*(x - y)/self.T)/(self.T**2)
            return -0.5*np.exp(-0.5*r2)*dr2dT
    
    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            raise NotImplementedError

class PerMatern12(PerKernel):
    """
        Periodic Matern12 kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        raise NotImplementedError
        super(PerMatern12,self).__init__()
        
    def f(self,x,y):
        normxy = np.sqrt(np.sum((x-y)**2))
        r = np.sin(np.pi*normxy/self.l)
        return np.exp(-r)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            normxy = np.sqrt(np.sum((x-y)**2))
            r = np.sin(np.pi*normxy/self.l)
            drdl = -np.pi*normxy*np.cos(np.pi*normxy/self.l)/(self.l**2)
            return -np.exp(-r)*drdl


class PerMatern32(PerKernel):
    """
        Periodic Matern32 kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        raise NotImplementedError
        super(PerMatern32,self).__init__()
        
    def f(self,x,y):
        normxy = np.sqrt(np.sum((x-y)**2))
        r = np.sin(np.pi*normxy/self.l)
        return (1 + SQRT3*r)*np.exp(-SQRT3*r)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            normxy = np.sqrt(np.sum((x-y)**2))
            r = np.sin(np.pi*normxy/self.l)
            drdl = -np.pi*normxy*np.cos(np.pi*normxy/self.l)/(self.l**2)
            return -3*r*np.exp(-SQRT3*r)*drdl


class PerMatern52(PerKernel):
    """
        Perodic Matern52 kernel. Number of hyperparams : 1
    """
    def __init__(self,dim):
        raise NotImplementedError
        super(PerMatern52,self).__init__()
    
    def f(self,x,y):
        normxy = np.sqrt(np.sum((x-y)**2))
        r = np.sin(np.pi*normxy/self.l)
        return (5.0*r**2/3 + SQRT5*r + 1)*np.exp(-SQRT5*r)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i == 0:
            normxy = np.sqrt(np.sum((x-y)**2))
            r = np.sin(np.pi*normxy/self.l)
            drdl = -np.pi*normxy*np.cos(np.pi*normxy/self.l)/(self.l**2)
            return -5.0*r*(SQRT5*r + 1)*np.exp(-SQRT5*r)/3*drdl

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
        self.nhyper = nout + (nout*(nout-1))//2
        self.hyperparams = None
        self.S = None
        self.W = None
        self.initialized = False
    
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        ls = hyperparams[:self.nout]
        thetas = hyperparams[self.nout:]
        S = np.zeros((self.nout,self.nout))
        S[0,0] = 1.0
        th_temp = thetas.copy()
        for i in range(1,self.nout):
            th_now,th_temp = th_temp[:i],th_temp[i:]
            s = utils.hypersphere_param(i+1,th_now)
            S[:i+1,i] = s
        W = np.matmul(S.transpose(),S) # S.T*S
        W = W*np.outer(ls,ls) # diag(l)*S.T*S*diag(l)
        self.S = S        
        self.W = W
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.S = None
        self.W = None
        self.initialized = False

    def f(self,x,y):
        return self.W[x,y]
    
    def df(self,x,y,i):
        ls = self.hyperparams[:self.nout]
        thetas = self.hyperparams[self.nout:]
        if i >= self.nout: #in respect to some theta hyperparameter
            j = i - self.nout + 1 #jth theta hyperparameter
            lj = utils.where_in_triangular(j) + 1 #column of jth hyperparam
            p = j - utils.triangular(lj - 2) #pth parital derivative of f_lj
            thetaslj = thetas[utils.triangular(lj-2):
                              utils.triangular(lj-1)] #thetas of f_lj
            dw = utils.hypersphere_param_derivative(lj,thetaslj,p)
            dw = utils.complete_with_zeros(dw,self.nout) #df_lj(thetaslj)/dp
            dS = np.zeros((self.nout,self.nout))
            dS[:,lj-1] = dw #dS/dthetaj
            dM = np.matmul(self.S.transpose(),dS)
            dM = dM.transpose() + dM #d(S'S)/dthetaj
            dW = dM*np.outer(ls,ls) #d(W)/dthetaj
            return dW[x,y]
        if i < self.nout: #in respect to some l hyperparameter
            dL = np.zeros((self.nout,self.nout))
            dL[:,i] = ls
            dL[i,:] = ls
            dL[i,i] = 2*ls[i]
            dW = np.matmul(self.S.transpose(),self.S)*dL
            return dW[x,y]

    def d2f(self,x,y,i):
        ls = self.hyperparams[:self.nout]
        thetas = self.hyperparams[self.nout:]
        if i >= self.nout: #in respect to some theta hyperparameter
            j = i - self.nout + 1 #jth theta hyperparameter
            lj = utils.where_in_triangular(j) + 1 #column of jth hyperparam
            p = j - utils.triangular(lj - 2) #pth parital derivative of f_lj
            thetaslj = thetas[utils.triangular(lj-2):
                              utils.triangular(lj-1)] #thetas of f_lj
            dw = utils.hypersphere_param_derivative(lj,thetaslj,p)
            dw = utils.complete_with_zeros(dw,self.nout) #df_lj(thetaslj)/dp
            dS = np.zeros((self.nout,self.nout))
            dS[:,lj-1] = dw #dS/dthetaj
            d2w = utils.hypersphere_param_derivative2(lj,thetaslj,p)
            d2w = utils.complete_with_zeros(dw,self.nout) #d2f_lj/dp
            d2S = np.zeros((self.nout,self.nout))
            d2S[:,lj-1] = d2w #d2S/dthetaj2
            d2M = np.matmul(self.S.transpose(),d2S)
            d2M = d2M.transpose() + d2M 
            d2M = d2M + 2*np.matmul(dS.transpose(),dS) #d2(M'M)/dthetaj
            d2W = d2M*np.outer(ls,ls) #d(W)/dthetaj
            return d2W[x,y]
        if i < self.nout: #in respect to some l hyperparameter
            dL = np.zeros((self.nout,self.nout))
            dL[i,i] = 2
            dW = np.matmul(self.S.transpose(),self.S)*dL
            return dW[x,y]

#==============================================================================
# Noise kernels
#==============================================================================
class NoiseKernel(Kernel):
    def __init__(self):
        pass


class IIDNoiseKernel(Kernel):
    """
        Regular noise kernel
    """
    def __init__(self):
        self.is_diagonal = True
        self.nhyper = 1
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.noise_var = hyperparams[0]
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.noise_var = None
        self.initialized = False
    
    def f(self,x,y):
        if x == y:
            return self.noise_var
        else:
            return 0.0
    
    def df(self,x,y,i):
        if i == 0:
            if x == y:
                return 1.0
            else:
                return 0.0
    
    def d2f(self,x,y,i):
        return 0.0


class MONoiseKernel(Kernel):
    """
        Noise kernel for multiple outputs. Inputs have 
        to be of the form [i,t]
    """
    def __init__(self,nout):
        self.nout = nout
        self.is_diagonal = True
        self.nhyper = nout
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.noise_vars = hyperparams
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.noise_vars = None
        self.initialized = False
    
    def f(self,x,y):
        if x == y:
            return self.noise_vars[x[0]]
        else:
            return 0.0
    
    def df(self,x,y,i):
        if x == y and x[0] == i:
            return 1.0
        else:
            return 0.0
    
    def d2f(self,x,y,i):
        return 0.0


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


class Sum(CompoundKernel):
    """
        Sum kernel. k(x,y) = k1(x,y) + k2(x,y)
        Number of hyperparams : k1.hyper + k2.hyper
    """
    def __init__(self,k1,k2):
        super(Sum,self).__init__(k1,k2)
        
    def f(self,x,y):
        return self.k1.f(x,y) + self.k2.f(x,y)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.df(x,y,i)
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k2.df(x,y,i)
    
    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.d2f(self,x,y,i)
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k2.df(x,y,i)

class Prod(CompoundKernel):
    """
        Hadamard product kernel. k(x,y) = k1(x,y)*k2(x,y)
        Number of hyperparams : k1.hyper + k2.hyper
    """
    def __init__(self,k1,k2):
        super(Prod,self).__init__(k1,k2)

    def f(self,x,y):
        return self.k1.f(x,y)*self.k2.f(x,y)
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.df(x,y,i)*self.k2.f(x,y)
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k1.f(x,y)*self.k2.df(x,y,i)
        
    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.d2f(x,y,i)*self.k2.f(x,y)
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k1.f(x,y)*self.k2.d2f(x,y,i)
    def dfdx(self,x,y):
        return self.k1.dfdx(x,y)*self.k2.f(x,y) + \
               self.k2.dfdx(x,y)*self.k1.f(x,y)


class DirectSum(CompoundKernel):
    """
        Direct sum kernel. k([x1,y1],[x2,y2]) = k1(x1,y1) + k2(x2,y2)
    """
    def __init__(self,k1,k2):
        super(DirectSum,self).__init__(k1,k2)
        
    def f(self,x,y):
        return self.k1.f(x[0],y[0]) + self.k2.f(x[1],y[1])
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.df(x[0],y[0],i)
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k2.df(x[1],y[1],i)

    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.d2f(x[0],y[0],i)
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k2.d2f(x[1],y[1],i)


class TensorProd(CompoundKernel):
    """
        Direct product kernel. k([x1,y1],[x2,y2]) = k1(x1,y1)*k2(x2,y2)
    """
    def __init__(self,k1,k2):
        super(TensorProd,self).__init__(k1,k2)
        
    def f(self,x,y):
        return self.k1.f(x[0],y[0])*self.k2.f(x[1],y[1])
    
    def df(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.df(x[0],y[0],i)*self.k2.f(x[1],y[1])
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k1.f(x[0],y[0])*self.k2.df(x[1],y[1],i)

    def d2f(self,x,y,i):
        assert i >= 0 and i < self.nhyper
        if i < self.k1.nhyper:
            return self.k1.d2f(x[0],y[0],i)*self.k2.f(x[1],y[1])
        elif i >= self.k1.nhyper:
            i = i - self.k1.nhyper
            return self.k1.f(x[0],y[0])*self.k2.d2f(x[1],y[1],i)