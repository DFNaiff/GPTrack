# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *
from .mo_kernels import SphericalCorr


class ShiftedMO(Kernel):
    """
        Variation of a compound kernel made specifically 
        for shifted multiple outputs time series.
        Initializes with [k1.hyperparameters]
    """
    def __init__(self,k,nout,dim):
        raise NotImplementedError
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

class ConvRBF1D(Kernel):
    def __init__(self,k,nout):
        raise NotImplementedError
    
    def f(self,x,y):
        px = self._lengthparams[x[:,0].long()]
        py = self._lengthparams[y[:,0].long()]
        sigmax = self._sigmaparams[x[:,0].long()]
        sigmay = self._sigmaparams[y[:,0].long()]
        first_term = sigmax*sigmay
        second_term = torch.pow(px*py,0.25)/torch.pow(px + py,0.5)
        third_term = torch.exp()
        r2 = torch.tensor(torch.sum((x - y)**2,1,keepdim=True) / \
            (self.hyperparams[0]**2))