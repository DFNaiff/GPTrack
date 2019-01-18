# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *

class CompositionKernel(Kernel):
    """
        Arguments: k1,g
        k(x,y) = k1(g(x),g(y))
    """
    def __init__(self,k,g):
        self.k = k
        self.g = g
        self.dim = g.dim
        self.nhyper = k.nhyper + g.nhyper
        self.hyperparams = None
        self.initialized = False
        self.positives = k.positives + g.positives
    
    def initialize(self,hyperparams):
        hyper1 = hyperparams[:self.k.nhyper]
        hyper2 = hyperparams[self.k.nhyper:]
        self.k.initialize(hyper1)
        self.g.initialize(hyper2)
        self.hyperparams = hyperparams
        self.initialized = True
    
    def reset(self):
        self.k.reset()
        self.g.reset()
        self.hyperparams = None
        self.initialized = False
        
    def f(self,x,y):
        return self.k.f(self.g.f(x),self.g.f(y))


class Warping(object):
    def __init__(self):
        pass
    
    def initialize(self,hyperparams):
        raise NotImplementedError


class KswamyWarping(Warping):
    """
        dim : dimension of input
        lb : either scalar or dim-dimensional tensor
        ub : either scalar or dim-dimensional tensor
    """
    def __init__(self,dim,lb=-0.0001,ub=1.0001):
        self.dim = dim
        self.nhyper = 2*dim
        self.lb = lb
        self.ub = ub
        self.initialized = False
        self.positives = [True]*self.nhyper
    
    def initialize(self,hyperparams):
        self.hyperparams = hyperparams
        self.alpha = torch.cat([h.unsqueeze(0) for 
                                h in self.hyperparams[:self.dim]])
        self.beta = torch.cat([h.unsqueeze(0) for 
                                h in self.hyperparams[self.dim:]])
    
    def reset(self):
        self.hyperparams = None
        self.initialized = False
    
    def f(self,x):
        xtilde = (x - self.lb)/(self.ub - self.lb)
        return 1 - torch.pow(1 - torch.pow(xtilde,self.alpha),self.beta)
    