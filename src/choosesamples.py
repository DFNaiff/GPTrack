# -*- coding: utf-8 -*-

import numpy as np
import torch

from . import gpobject
from . import utils


def choose_samples_grid(data,kernel,noisekernel,
                        positives,hparams,**kwargs):
    """
        data : (xdata,ydata) tuple
        kernel : kernel
        noisekernel : noisekernel
        positives : [bool]*(kernel.nhyper + noisekernel.nhyper)
                     list of positive hyperparameters
        hparams : initial hyperparameters
        center_first : whether to find MLE first
        kwargs accepts gp.optimize parameters (with option = "B")
    """
    assert len(positives) == len(hparams)
    center_first = kwargs.get("center_first",True)
    alpha = kwargs.get("alpha",True)
    nmax = kwargs.get("nmax",True)
    default_prior_variance = kwargs.get("default_prior_variances",1.0)
    verbose = kwargs.get("verbose",0)
    if center_first:
        if verbose: print("Centering")
        gp = gpobject.GPObject(kernel,noisekernel,hparams,data)
        gp = gp.optimize(option="B",**kwargs)
        hparams = gp.hparams
    else:
        hparams = [torch.tensor(p) for p in hparams]
    if verbose >= 2:
        print("Parameters of center:")
        print(hparams)
    #Extremely inneficient to make ANOTHER gp here
    #Adjust hparams so we can get second derivatives
    gp,hparams_new,hparams_scalar = _set_gp_with_grad(kernel,noisekernel,
                                                      data,hparams,positives)
    #Get second derivatives
    if verbose >= 1:
        print("Getting second derivatives")
    grads2 = _get_grads2(gp,hparams_new)
    if verbose >= 2:
        print(grads2)
    deltas = []
    prior_variances = []
    #Sample
    for i,grad2 in enumerate(grads2):
        if grad2 >= 0: #Some error in grad2 calculation. Abort
            deltas.append(0.0)
            prior_variances.append(default_prior_variance)
        else:
            sigma2 = -1/grad2
            deltas.append(np.sqrt(-sigma2*2*np.log(alpha)))
            prior_variances.append(sigma2)
    if verbose >= 2:
        print(prior_variances)
        print(deltas)
    hparamssamples = _laplace_grid_sampling(hparams_scalar,deltas,
                                            positives,nmax)
    # TODO : Better final sampling strategy
    return hparamssamples,prior_variances


def choose_samples_laplace_a(data,kernel,noisekernel,
                             positives,hparams,**kwargs):
    """
        data : (xdata,ydata) tuple
        kernel : kernel
        noisekernel : noisekernel
        positives : [bool]*(kernel.nhyper + noisekernel.nhyper)
                     list of positive hyperparameters
        hparams : initial hyperparameters
        center_first : whether to find MLE first
        kwargs accepts gp.optimize parameters (with option = "B")
    """
    assert len(positives) == len(hparams)
    center_first = kwargs.get("center_first",True)
    num_samples = kwargs.get("num_samples",True)
    default_prior_variance = kwargs.get("default_prior_variances",1.0)
    verbose = kwargs.get("verbose",0)
    num_params = len(hparams)
    if center_first:
        if verbose: print("Centering")
        gp = gpobject.GPObject(kernel,noisekernel,hparams,data)
        gp = gp.optimize(option="B",**kwargs)
        hparams = gp.hparams
    else:
        hparams = [torch.tensor(p) for p in hparams]
    if verbose >= 2:
        print("Parameters of center:")
        print(hparams)
    #Extremely inneficient to make ANOTHER gp here
    #Adjust hparams so we can get second derivatives
    gp,hparams_new,hparams_scalar = _set_gp_with_grad(kernel,noisekernel,
                                                      hparams,positives)
    #Get second derivatives
    if verbose >= 1:
        print("Getting hessian")
    hessian = _get_hessian(gp,hparams_new,num_params)
    if verbose >= 2:
        print(hessian)
    prior_variances = []
    #Sample
    for i,_ in enumerate(hparams_scalar):
        if hessian[i][i] >= 0: #Some error in grad2 calculation. Abort
            prior_variances.append(default_prior_variance)
        else:
            sigma2 = -1/hessian[i][i]
            prior_variances.append(sigma2)
    if verbose >= 2:
        print(prior_variances)
    hparamssamples = np.random.multivariate_normal(mean=np.zeros(num_params),
                                                   cov=hessian,
                                                   size=num_samples)
    return hparamssamples,prior_variances


def _get_grads2(gp,hparams):
    grads2 = []
    dl = torch.autograd.grad(gp.loglikelihood,hparams,
                             create_graph=True)
    for i,p in enumerate(hparams):
        d2li = torch.autograd.grad(dl[i],p,allow_unused=True,
                                   retain_graph = True)[0]
        if type(d2li) == type(None):
            grads2.append(0.0)
        else:
            grads2.append(d2li.item())
    return np.array(grads2)


def _get_hessian(gp,hparams,num_params):
    #TODO : Far to inefficient
    hessian = [[None]*num_params]*num_params
    dl = torch.autograd.grad(gp.loglikelihood,hparams,
                             create_graph=True)
    for i,_ in enumerate(hparams):
        for j,pj in enumerate(hparams):
            d2lij = torch.autograd.grad(dl[i],pj,allow_unused=True,
                                        retain_graph = True)[0]
            if type(d2lij) == type(None):
                hessian[i][j] = 0.0
            else:
                hessian[i][j] = d2lij.item()
    return np.array(hessian)


def _laplace_grid_sampling(hparams_scalar,deltas,positives,nmax):
    deltamax = max(deltas)
    hparamssamples = []
    for i,hparam in enumerate(hparams_scalar):
        ni = utils.roundodd(nmax*deltas[i]/deltamax)
        if positives[i]:
            p1 = np.logspace(np.log(hparam) - deltas[i],
                             np.log(hparam),
                             ni//2 + 1,base=np.e)[:-1]
            p2 = np.logspace(np.log(hparam),
                             np.log(hparam) + deltas[i],
                             ni//2 + 1,base=np.e)
        else:
            p1 = np.linspace(hparam - deltas[i],
                             hparam,
                             ni//2 + 1)[:-1]
            p2 = np.linspace(hparam,
                             hparam + deltas[i],
                             ni//2 + 1)
        hparamssample = np.hstack([p1,p2])
        hparamssamples.append(hparamssample)
    return hparamssamples


def _set_gp_with_grad(kernel,noisekernel,data,hparams,positives):
    #hparams_new,hparams_scalar returns the LOGARITHM of 
    #hyperparameters, if positives[i]
    hparams_new = [None]*len(hparams)
    hparams_feed = [None]*len(hparams)
    for i,hparam in enumerate(hparams):
        hparam_new = hparam.clone()
        if positives[i]: #Sample from logarithm
            hparam_new = torch.log(hparam).clone()
        hparam_new.requires_grad_() #Allow backprop
        hparams_new[i] = hparam_new
        if positives[i]: #Go back to exp in feeding GP
            hparams_feed[i] = torch.exp(hparam_new)
        else:
            hparams_feed[i] = hparam_new.clone()
    gp = gpobject.GPObject(kernel,noisekernel,hparams_feed,data)
    hparams_scalar = [hp.item() for hp in gp.hparams]
    return gp,hparams_new,hparams_scalar
#def choose_samples_langevin(data,kernel,noisekernel,
#                        sampleables,positives,phi0,
#                        center_first = True,
#                        trans = None,
#                        adjustables = True,
#                        tol = 1e-4,
#                        method = "L-BFGS-B",
#                        nsamples = 60,
#                        epsilon = 0.1):
#    assert len(positives) == len(phi0)
#    #TODO : DO NOT IGNORE PRIORS (MAP, not MLE)
#    gp = gpobject.GPObject(kernel,noisekernel,phi0,data)
#    if center_first:
#        if adjustables == True:
#            adjustables = [True]*len(positives)
#        else:
#            for i,adjustable in enumerate(adjustables): #Enforcing constraint
#                if not adjustable:
#                    sampleables[i] = False
#        gp,phi0_hmc = gp.optimize(adjustables,positives,trans=trans,
#                                verbose=2,tol=tol,method=method,
#                                rethypers=True)
#    else:
#        phi0_hmc = phi0        
#    inds = [i for i in range(len(sampleables)) if sampleables[i]]
#    #Langevin monte carlo
#    phisamples = []
#    energies = []
#    #With transformations one must be careful
#    dim = len(phi0_hmc)
#    phi_hmc = phi0.copy()
#    phi_hmc_trans = _transform(phi_hmc,trans)
#    phisamples.append(phi_hmc)
#    energy = -gp.loglikelihood
#    energies.append(energy)
#    genergy = np.zeros(dim)
#    genergy_samples = -gp._dbatch_loglikelihood(inds)
#    genergy[inds] = genergy_samples
#    genergy_trans = _transform_grads(genergy,phi_hmc,trans)
#    while len(phisamples) < nsamples:
#        print(len(phisamples))
#        p = np.random.randn(dim)
#        hamiltonian = np.sum(p**2)/2.0 + energy
#        p = p - epsilon*genergy_trans/2
#        #new sample
#        phi_new_trans = phi_hmc_trans + epsilon*p
#        phi_new = _untransform(phi_new_trans,trans)
#        print(phi_new)
#        #energy and grad_energy
#        gp_new = gpobject.GPObject(kernel,noisekernel,phi_new,data)
#        energy_new = -gp_new.loglikelihood
#        genergy_new = np.zeros(dim)
#        genergy_new_samples = -gp._dbatch_loglikelihood(inds)
#        genergy_new[inds] = genergy_new_samples
#        genergy_trans_new = _transform_grads(genergy_new,phi_new,trans)
#        
#        p = p - epsilon*genergy_trans_new/2
#        hamiltonian_new = np.sum(p**2)/2.0 + energy_new
#        deltahamiltonian = hamiltonian_new - hamiltonian
#        print(deltahamiltonian)
#        print('---')
#        if deltahamiltonian < 0:
#            accept = True
#        elif np.random.random() < np.exp(-deltahamiltonian):
#            accept = True
#        else:
#            accept = False
#        if accept:
#            gp = gp_new
#            genergy_trans = genergy_trans_new
#            phi_hmc = phi_new
#            phi_hmc_trans = phi_new_trans
#            energy = energy_new
#            phisamples.append(phi_hmc)
#            energies.append(energy)
#    return phisamples,energies
#
#
#def get_prior_lengthscales(data,dim):
#    X = np.array(data[0])
#    Y = np.array(data[1]).reshape(-1,1)
#    gpykernel = GPy.kern.RBF(input_dim=dim,
#                             lengthscale = [1.0]*dim,
#                             ARD = True)
#    gpymodel = GPy.models.GPRegression(X,Y,gpykernel)
#    gpymodel.likelihood.variance = 1e-6
#    gpymodel.optimize(messages=True)
#    return gpymodel.kern.lengthscale.values
#    
#
#def _transform(phi,trans):
#    phitrans = phi.copy()
#    for i,_ in enumerate(phitrans): #transformations
#        if trans[i] == "log":
#            phitrans[i] = np.log(phitrans[i])
#        if trans[i] == "sqrt":
#            phitrans[i] = np.sqrt(phitrans[i])
#    return phitrans
#   
#
#def _untransform(phi,trans):
#    phitrans = phi.copy()
#    for i,_ in enumerate(phitrans): #transformations
#        if trans[i] == "log":
#            phitrans[i] = np.exp(phitrans[i])
#        if trans[i] == "sqrt":
#            phitrans[i] = np.square(phitrans[i])
#    return phitrans
#
#
#def _transform_grads(dg,phi,trans):
#    for i,_ in enumerate(dg): #transformations
#        if trans[i] == "log":
#            dg[i] = phi[i]*dg[i]
#        if trans[i] == "sqrt":
#            dg[i] = 2*phi[i]*dg[i] #TODO : Is this the correct way?
#    return dg