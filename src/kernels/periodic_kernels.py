# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *


class PerKernel(Kernel):
    def __init__(self):
        self.dim = 1
        self.nhyper = 2
        self.hyperparams = None
        self.initialized = False
        self.positives = [True,True]
        
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
