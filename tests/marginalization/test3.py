# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import emcee
import torch

from src import kernels,utilstorch,gpobject

SQRT2PI = float(np.sqrt(2*np.pi))

def emcee_sampler(kernel,noisekernel,data,
                  nwalkers,prior_lnprob):
    def lnprob(sample):
        posinds = kernel.positives + noisekernel.positives
        sample[posinds] = np.exp(sample[posinds])
        hparams = [torch.tensor(s) for s in sample]
        gp = gpobject.GPObject(kernel,noisekernel,hparams,data)
        likelihood = gp.loglikelihood
        likelihood += prior_lnprob(sample)
        return likelihood
    ndim = kernel.nhyper + noisekernel.nhyper
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    pos,prob,state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos,100)
    print(pos)
        
#class GPJoiner(object):
#    def __init__(self,gplist):
#        self.gplist = gplist
#    
#    def predict_mean(self,x):
#        means = [gp.predict_batch(x)[0] for gp in self.gplist]
#        means = np.hstack(means)
#        mean = np.mean(means,axis=1)
#        return mean
#


#==============================================================================
# OTHER THING
#==============================================================================
sigma = 3.0
def prior_sampler(N):
    return list(np.exp(sigma*np.random.normal(size=(N,3))))

def prior_logpdf(sample):
    return np.sum(-np.log(sample)**2/(2*sigma**2) - np.log(sample*sigma*SQRT2PI))


loadfile = np.load("sine_data_1.npz")
X = loadfile["X"].reshape(-1,1)
Y = loadfile["Y"].reshape(-1,1)

kernel = kernels.Constant()*kernels.IsoRBF(dim=1)
noisekernel = kernels.IIDNoiseKernel()
hparams = np.array([0.5,0.2,0.05])
bounds = {0:[1e-6,None],
          1:[1e-6,None],
          2:[1e-6,None]}
warpings = {0:'sqrt',1:'sqrt',2:'sqrt'}

gpsamples = emcee_sampler(kernel,noisekernel,(X,Y),100,prior_logpdf)
#gpjoiner = GPJoiner(gpsamples)
#gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
#gp = gp.optimize(verbose=1,opt_choice="cg",positives_default="bound",
#                 beta_update_fn="FR",line_search_fn="goldstein")
##
#xpred = np.linspace(0,1).reshape(-1,1)
#ypred,_ = gp.predict_batch(xpred)
#ypredmrg = gpjoiner.predict_mean(xpred)
##
#plt.plot(X,Y,'b*')
#plt.plot(xpred,ypred,'g')
#plt.plot(xpred,ypredmrg,'m')
#plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')