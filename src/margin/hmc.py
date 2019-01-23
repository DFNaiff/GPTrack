# -*- coding: utf-8 -*-

import numpy as np
import torch

from ..gpobject import GPObject

def adjust_hparams(hparams,warping_list):
    hparams_feed = [None]*len(hparams)
    for i,hparam in enumerate(hparams):
        hp_t = torch.tensor(hparam,requires_grad=True)
        if warping_list[i]:
            hparams_feed[i] = torch.exp(hp_t)
        else:
            hparams_feed[i] = hp_t.clone()
        hparams_feed[i].requires_grad_()
    return hparams_feed
    
def hmc_samples(kernel,noisekernel,data,hparams0,
                numsamples,stepsize=0.1,lsteps=10,
                **kwargs):
    """
        kernel : kernel
        noise_kernel : noise kernel
        data : (x,y) pair
        hparams0 : initial sample (array)
        numsamples : number of samples
        stepsize : step size for hmc
        lsteps : number of leapfrog steps
        
        Warping only implemented for logarithmic data
    """
    positives_default = kwargs.get("positives_default","warp")
    warp_dict = kwargs.get("warpings",dict())
    bounds_dict = kwargs.get("bounds",dict())
    plb = kwargs.get("plb",1e-10)
    verbose = kwargs.get("verbose",0)
    
    dimsamples = len(hparams0)

    if not (kernel.has_prior() and noisekernel.has_prior()):
        raise ValueError("You must initialize kernel priors")
    
    def u_and_grad_u(q):
        hparams_feed = adjust_hparams(q,warping_list)
        gpnew = GPObject(kernel,noisekernel,hparams_feed,data)
        u = -gpnew.logposteriori()
        grad_u = np.array(torch.autograd.grad(u,hparams_feed))
        return u,grad_u
    
    #Make warping list
    warping_list = np.array([False]*(kernel.nhyper + noisekernel.nhyper))
    warping_list[list(warp_dict.keys())] = True
    if positives_default == "warp":
        positive_list = np.array(kernel.positives + noisekernel.positives)
        warping_list = warping_list | positive_list
    hparams0[warping_list] = np.log(hparams0[warping_list]) #For now warping
    sample_list = np.zeros((numsamples+1,dimsamples))
    sample_list[0,:] = hparams0
    #HMC algorithm
    for j in range(1,numsamples+1):
        hparams = hparams0.copy()   
        p = np.random.randn(dimsamples)
        p0 = p.copy()
        u0,grad_u = u_and_grad_u(hparams)
        p = p - stepsize*grad_u/2
        for i in range(lsteps): #Leapfrog steps
            hparams = hparams + stepsize*p
            u,grad_u = u_and_grad_u(hparams)
            if i != lsteps-1: #Full step
                p = p - stepsize*grad_u
            else: #Half step
                p = p - stepsize*grad_u/2
        p = -p #Negate momentum
        dH = (u.detach().numpy() + (p**2).sum()/2) - \
             (u0.detach().numpy() + (p0**2).sum()/2)
        if dH < 0 or np.random.rand() < np.exp(-dH): #Accept
            sample_list[j,:] = hparams
            hparams0 = hparams
        else: #Reject
            sample_list[j,:] = hparams0
        if verbose >= 1:
            if j%50 == 0:
                print("%i samples completed"%(j+1))
    samples = np.array(sample_list)
    samples[:,warping_list] = np.exp(samples[:,warping_list])
    return samples
