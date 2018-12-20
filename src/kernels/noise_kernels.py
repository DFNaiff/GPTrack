# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *


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
