# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import copy
import functools

import numpy as np
import torch

from . import gpobject
from . import utils
from . import utilsla
from . import utilstorch

def optimize(kernel,noisekernel,hparams,
             data,positives,adjustable=True,
                  num_starts = 1,verbose=False,
                  penalize_runtime_error = True):
    """
        Choose new parameters for the GP based on 
        MLE estimation.
        input:
            positives : [bool] list of positive parameters
            adjustables : [bool] list of parameters to change
            num_starts : Number of times to run L-BFGS
        returns:
            GPObject with new parameters. If rethyper also hyperparameters
    """
    #TODO : Check
    _param_opt = functools.partial(_optimize_single_start,
                    kernel = kernel,noisekernel = noisekernel,data=data,
                    positives = positives,adjustable = adjustable,
                    verbose = verbose, 
                    penalize_runtime_error = penalize_runtime_error)
    nll,hparams_new = _param_opt(hparams = hparams)
    for i in range(1,num_starts):
        hparams_test = _perturb(hparams,positives)
        nll_test,hparams_test = _param_opt(hparams = hparams_test)
        if nll_test < nll:
            hparams_new = hparams_test
        print(nll_test,hparams_new)
    gpnew = gpobject.GPObject(kernel,noisekernel,hparams_new,
                              data)
    return gpnew

def _optimize_single_start(kernel,noisekernel,hparams,
                           data,positives,adjustable=True,
                           verbose=False,
                           penalize_runtime_error = True):
    xdata,ydata = data
    for i,_ in enumerate(hparams): #Convert to tensor
        hparams[i] = torch.tensor(hparams[i])
    
    def _negative_log_likelihood(hparams,positives):
        hparams_feed = [None]*len(hparams)
        for i,_ in enumerate(hparams):
            if positives[i]:
                hparams_feed[i] = hparams[i]**2
            else:
                hparams_feed[i] = hparams[i].clone()
        if verbose:
            print([h.item() for h in hparams_feed])
        try:
            gpnew = gpobject.GPObject(kernel,noisekernel,hparams_feed,
                             (xdata,ydata))
            result = -gpnew.loglikelihood
        except RuntimeError:
            result = 1e12 + sum(hparams_feed)
        if verbose:
            print(result)
            print("-"*10)
        return result
    
    if adjustable == True:
        adjustable = True*len(positives)
    #Adjust hparams so we can differentiate
    hparams_new = []
    for i,hparam in enumerate(hparams):
        #TODO : Put adjustable here
        hparam_new = hparam.clone()
        if positives[i]: #TODO : Check
            hparam_new = torch.sqrt(hparam).clone()
        hparam_new.requires_grad_()
        hparams_new.append(hparam_new)
    #Optmizer
    optimizer = torch.optim.LBFGS(hparams_new,max_iter=100)
    optimizer.zero_grad()
    def closure():
        optimizer.zero_grad()
        nll = _negative_log_likelihood(hparams_new,positives)
        nll.backward(retain_graph=True)
        return nll
    nll = optimizer.step(closure)
    #Create new gp
    for i,_ in enumerate(hparams_new):
        hparams_new[i].requires_grad = False
        if positives[i]:
            hparams_new[i] = hparams_new[i]**2
    return nll.item(),hparams_new


def _perturb(hparams,positives,noise = 1.0):
    hparams_perturb = [None]*len(hparams)
    for i,p in enumerate(hparams):
        if positives[i]:
            hparams_perturb[i] = (np.sqrt(p) + noise*np.random.randn())**2
        else:
            hparams_perturb[i] = p + noise*np.random.randn()
    return hparams_perturb