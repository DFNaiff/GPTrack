# -*- coding: utf-8 -*-
import numpy as np
import torch

from .base import *
from .constants import *
from .. import utils
from .. import utilstorch


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
        U = torch.zeros((self.nout,self.nout))
        for i in range(1,self.nout+1):
            start_ind = utils.triangular(i-1)
            end_ind = utils.triangular(i)
            diagtensor = torch.tensor(hyperparams[start_ind:end_ind])
            U += torch.diag(diagtensor,self.nout-i)
        W = torch.matmul(U.transpose(1,0),U) # U.T*U
        self.W = W
        self.initialized = True
    
    def reset(self):
        self.hyperparams = None
        self.W = None
        self.initialized = False

    def f(self,x,y):
        return self.W[x.long(),y.long()]
