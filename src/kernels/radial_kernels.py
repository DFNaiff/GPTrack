# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *


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


class IsoRationalQuadratic(IsoRadKernel):
    """
        Isotropic Matern52 kernel. Number of hyperparams : 1
        {0:'l',1:'alpha'}
    """
    def __init__(self,dim):
        super(IsoRationalQuadratic,self).__init__(dim)
        self.nhyper = 2
        
    def f(self,x,y):
        r2 = torch.tensor(torch.sum((x - y)**2,1,keepdim=True) / \
                (self.hyperparams[0]**2))
        return (1 + r2/(2*self.hyperparams[1]))**(-self.hyperparams[1])

class ARDRadKernel(Kernel):
    def __init__(self,dim):
        self.dim = dim
        self.nhyper = dim
        self.hyperparams = None
        self.initialized = False
        
    def initialize(self,hyperparams):
        assert len(hyperparams) == self.nhyper
        self.hyperparams = hyperparams
        self.length_scales = torch.cat([h.unsqueeze(0)
                                        for h in self.hyperparams])
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False
    

class ARDRBF(ARDRadKernel):
    """
        Isotropic RBF kernel. Number of hyperparams : 1
        {0:'l'}
    """
    def __init__(self,dim):
        super(ARDRBF,self).__init__(dim)
        
    def f(self,x,y):
        r2 = torch.tensor(torch.sum((x - y)**2/(self.length_scales**2),1,keepdim=True))
        return torch.exp(-0.5*r2)


#class IsoMatern12(IsoRadKernel):
#    """
#        Isotropic Matern12 kernel. Number of hyperparams : 1
#        {0:'l'}
#    """
#    def __init__(self,dim):
#        super(IsoMatern12,self).__init__(dim)
#        
#    def f(self,x,y):
#        r = torch.tensor(torch.sqrt(torch.sum((x - y)**2,1,keepdim=True)) / \
#                self.hyperparams[0])
#        return torch.exp(-r)
#
#
#class IsoMatern32(IsoRadKernel):
#    """
#        Isotropic Matern52 kernel. Number of hyperparams : 1
#        {0:'l'}
#    """
#    def __init__(self,dim):
#        super(IsoMatern32,self).__init__(dim)
#        
#    def f(self,x,y):
#        r = torch.tensor(torch.sqrt(torch.sum((x - y)**2,1,keepdim=True)) / \
#                self.hyperparams[0])
#        return (1 + SQRT3*r)*torch.exp(-SQRT3*r)
#        
#
#class IsoMatern52(IsoRadKernel):
#    """
#        Isotropic Matern52 kernel. Number of hyperparams : 1
#        {0:'l'}
#    """
#    def __init__(self,dim):
#        super(IsoMatern52,self).__init__(dim)
#        
#    def f(self,x,y):
#        r = torch.tensor(torch.sqrt(torch.sum((x - y)**2,1,keepdim=True)) / \
#                self.hyperparams[0])
#        return (1 + SQRT5*r + 5.0/3*r**2)*torch.exp(-SQRT5*r)
#
#
#class IsoRationalQuadratic(IsoRadKernel):
#    """
#        Isotropic Matern52 kernel. Number of hyperparams : 1
#        {0:'l',1:'alpha'}
#    """
#    def __init__(self,dim):
#        super(IsoRationalQuadratic,self).__init__(dim)
#        self.nhyper = 2
#        
#    def f(self,x,y):
#        r2 = torch.tensor(torch.sum((x - y)**2,1,keepdim=True) / \
#                (self.hyperparams[0]**2))
#        return (1 + r2/(2*self.hyperparams[1]))**(-self.hyperparams[1])
