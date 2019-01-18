# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *


class SpectralGaussian(Kernel):
    """
        Spectral gaussian kernel. Number of hyperparams : 2
        {0:'l',1:'mu'}
    """
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
    
    def f(self,x,y):
        r = torch.sqrt(torch.sum((x - y)**2,1,keepdim=True))
        return torch.exp(-0.5*r**2/(self.hyperparams[0]**2))*\
               torch.cos(self.hyperparams[1]*r)
