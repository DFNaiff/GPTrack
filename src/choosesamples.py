# -*- coding: utf-8 -*-

import numpy as np
import GPy

from . import gpobject
from . import utils

def choose_samples_grid(data,kernel,noisekernel,
                        sampleables,positives,phi0,
                        center_first = True,
                        trans = None,
                        adjustables = True,
                        tol = 1e-4,
                        method = "L-BFGS-B",
                        alpha = 0.3,
                        nmax = 5,
                        default_prior_variance = 1.0):
    """
        data : [xdata,zdata] list
        kernel : kernel
        noisekernel : noisekernel
        sampleables : [bool]*(kernel.nhyper + noisekernel.nhyper) 
                              list of sampleables hyperparams
        positives : [bool]*(kernel.nhyper + noisekernel.nhyper)
                            list of positive hyperparams
        phi0 : initial hyperparameter guess
        center_first : whether to find MLE first. If False 
                       then phi0 must already be the MLE.
        adjustables : True or [bool]*(kernel.nhyper + noisekernel.nhyper)
                      if True all hyperparameters are to be adjusted. 
                      else is a list of parameters to be adjusted.
                      ANY ITEM False IN ADJUSTABLES MUST BE FALSE 
                      IN SAMPLEABLES.
        trans : list of transformation to 
                be applied on ith parameters on optimization. 
                options (NOT ON SAMPLING)
                options: None,"log","sqrt"
                default: [None]        
        tol : solver tolerance.
        method : "L-BFGS-B","TNC" or "SLSQP"
        alpha : default : 0.3
        nmax : default : 5
        default_prior_variance: default: 1.0
        Returns:
            (phisamples,prior_variances) to be used on GPTrack
    """
    #TODO : DO NOT IGNORE PRIORS (MAP, not MLE)
    assert len(positives) == len(phi0)

    gp = gpobject.GPObject(kernel,noisekernel,phi0,data)
    if center_first:
        if adjustables == True:
            adjustables = [True]*len(positives)
        else:
            for i,adjustable in enumerate(adjustables): #Enforcing constraint
                if not adjustable:
                    sampleables[i] = False
        gp,hypers = gp.optimize(adjustables,positives,trans=trans,
                                verbose=2,tol=tol,method=method,
                                rethypers=True)
    else:
        hypers = phi0        
    inds = [i for i in range(len(sampleables)) if sampleables[i]]
    #Variances calculation (Laplace's method)
    grads2 = gp._d2batch_loglikelihood(inds)
    deltas = []
    prior_variances = []
    for i,sampleable in enumerate(sampleables):
        if sampleable:
            grad2i = utils.equals_before(sampleables,i)
            grad2 = grads2[grad2i] # FIXME : More elegant solution
            if grad2 >= 0: #Some error in grad2 calculation. Abort
                deltas.append(0.0)
                prior_variances.append(default_prior_variance)
            else:
                if positives[i]:
                    sigma2 = -1/(grad2*hypers[i])
                else:
                    sigma2 = -1/grad2
                deltas.append(np.sqrt(-sigma2*2*np.log(alpha)))
                prior_variances.append(sigma2)
        else:
            deltas.append(0.0)
            prior_variances.append(default_prior_variance)
    # TODO : Better final sampling strategy
    deltamax = max(deltas)
    phisamples = []
    for i,sampleable in enumerate(sampleables):
        ni = utils.roundodd(nmax*deltas[i]/deltamax)
        if positives[i]:
            p1 = np.logspace(np.log(hypers[i]) - deltas[i],np.log(hypers[i]),
                             ni//2 + 1,base=np.e)[:-1]
            p2 = np.logspace(np.log(hypers[i]),np.log(hypers[i]) + deltas[i],
                             ni//2 + 1,base=np.e)
        else:
            p1 = np.linspace(hypers[i] - deltas[i],hypers[i],ni//2 + 1)[:-1]
            p2 = np.linspace(hypers[i],hypers[i] + deltas[i],ni//2 + 1)
        phisample = np.hstack([p1,p2])
        phisamples.append(phisample)
    return phisamples,prior_variances


def choose_samples_langevin(data,kernel,noisekernel,
                        sampleables,positives,phi0,
                        center_first = True,
                        trans = None,
                        adjustables = True,
                        tol = 1e-4,
                        method = "L-BFGS-B",
                        nsamples = 60,
                        epsilon = 0.1):
    assert len(positives) == len(phi0)
    #TODO : DO NOT IGNORE PRIORS (MAP, not MLE)
    gp = gpobject.GPObject(kernel,noisekernel,phi0,data)
    if center_first:
        if adjustables == True:
            adjustables = [True]*len(positives)
        else:
            for i,adjustable in enumerate(adjustables): #Enforcing constraint
                if not adjustable:
                    sampleables[i] = False
        gp,phi0_hmc = gp.optimize(adjustables,positives,trans=trans,
                                verbose=2,tol=tol,method=method,
                                rethypers=True)
    else:
        phi0_hmc = phi0        
    inds = [i for i in range(len(sampleables)) if sampleables[i]]
    #Langevin monte carlo
    phisamples = []
    energies = []
    #With transformations one must be careful
    dim = len(phi0_hmc)
    phi_hmc = phi0.copy()
    phi_hmc_trans = _transform(phi_hmc,trans)
    phisamples.append(phi_hmc)
    energy = -gp.loglikelihood
    energies.append(energy)
    genergy = np.zeros(dim)
    genergy_samples = -gp._dbatch_loglikelihood(inds)
    genergy[inds] = genergy_samples
    genergy_trans = _transform_grads(genergy,phi_hmc,trans)
    while len(phisamples) < nsamples:
        print(len(phisamples))
        p = np.random.randn(dim)
        hamiltonian = np.sum(p**2)/2.0 + energy
        p = p - epsilon*genergy_trans/2
        #new sample
        phi_new_trans = phi_hmc_trans + epsilon*p
        phi_new = _untransform(phi_new_trans,trans)
        print(phi_new)
        #energy and grad_energy
        gp_new = gpobject.GPObject(kernel,noisekernel,phi_new,data)
        energy_new = -gp_new.loglikelihood
        genergy_new = np.zeros(dim)
        genergy_new_samples = -gp._dbatch_loglikelihood(inds)
        genergy_new[inds] = genergy_new_samples
        genergy_trans_new = _transform_grads(genergy_new,phi_new,trans)
        
        p = p - epsilon*genergy_trans_new/2
        hamiltonian_new = np.sum(p**2)/2.0 + energy_new
        deltahamiltonian = hamiltonian_new - hamiltonian
        print(deltahamiltonian)
        print('---')
        if deltahamiltonian < 0:
            accept = True
        elif np.random.random() < np.exp(-deltahamiltonian):
            accept = True
        else:
            accept = False
        if accept:
            gp = gp_new
            genergy_trans = genergy_trans_new
            phi_hmc = phi_new
            phi_hmc_trans = phi_new_trans
            energy = energy_new
            phisamples.append(phi_hmc)
            energies.append(energy)
    return phisamples,energies


def get_prior_lengthscales(data,dim):
    X = np.array(data[0])
    Y = np.array(data[1]).reshape(-1,1)
    gpykernel = GPy.kern.RBF(input_dim=dim,
                             lengthscale = [1.0]*dim,
                             ARD = True)
    gpymodel = GPy.models.GPRegression(X,Y,gpykernel)
    gpymodel.likelihood.variance = 1e-6
    gpymodel.optimize(messages=True)
    return gpymodel.kern.lengthscale.values
    

def _transform(phi,trans):
    phitrans = phi.copy()
    for i,_ in enumerate(phitrans): #transformations
        if trans[i] == "log":
            phitrans[i] = np.log(phitrans[i])
        if trans[i] == "sqrt":
            phitrans[i] = np.sqrt(phitrans[i])
    return phitrans
   

def _untransform(phi,trans):
    phitrans = phi.copy()
    for i,_ in enumerate(phitrans): #transformations
        if trans[i] == "log":
            phitrans[i] = np.exp(phitrans[i])
        if trans[i] == "sqrt":
            phitrans[i] = np.square(phitrans[i])
    return phitrans


def _transform_grads(dg,phi,trans):
    for i,_ in enumerate(dg): #transformations
        if trans[i] == "log":
            dg[i] = phi[i]*dg[i]
        if trans[i] == "sqrt":
            dg[i] = 2*phi[i]*dg[i] #TODO : Is this the correct way?
    return dg