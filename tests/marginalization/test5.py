# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import torch
import emcee

from src import kernels,utilstorch,gpobject

SQRT2PI = float(np.sqrt(2*np.pi))

def adjust_hparams(hparams,positive_list):
    hparams_feed = [None]*len(hparams)
    for i,hparam in enumerate(hparams):
        hp_t = torch.tensor(hparam,requires_grad=True)
        if positive_list[i]:
            hparams_feed[i] = torch.exp(hp_t)
        else:
            hparams_feed[i] = hp_t.clone()
        hparams_feed[i].requires_grad_()
    return hparams_feed
    
def hmc_samples(kernel,noise_kernel,data,hparams0,
                numsamples,stepsize,lsteps):
    """
        kernel : kernel
        noise_kernel : noise kernel
        data : (x,y) pair
        hparams0 : initial sample (array)
        numsamples : number of samples
        stepsize : step size for hmc
        lsteps : number of leapfrog steps
    """
    #Later add frozen and whatever
    def u_and_grad_u(q):
        hparams_feed = adjust_hparams(q,positive_list)
        gpnew = gpobject.GPObject(kernel,noisekernel,hparams_feed,data)
        u = -gpnew.logposteriori()
        grad_u = np.array(torch.autograd.grad(u,hparams_feed))
        return u,grad_u
        
    kernel.set_priors()
    noise_kernel.set_priors()
    positive_list = kernel.positives + noisekernel.positives
    sample_list = []
    hparams0[positive_list] = np.log(hparams0[positive_list]) #For now warping
    sample_list.append(hparams0)
    dimsamples = len(hparams0)
    #HMC algorithm
    p = np.random.randn(dimsamples)
    for _ in range(numsamples):
        hparams = hparams0.copy()   
#        p = np.random.randn(dimsamples)
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
            sample_list.append(hparams)
            hparams0 = hparams
        else: #Reject
            sample_list.append(hparams0)
        if (_)%50 == 0:
            print(_)

    samples = np.array(sample_list)
    samples[:,positive_list] = np.exp(samples[:,positive_list])
    return samples


class GPJoiner(object):
    def __init__(self,kernel,noise_kernel,gpsamples,data):
        self.gplist = []
        for i,sample in enumerate(gpsamples):
            gp = gpobject.GPObject(kernel,noise_kernel,sample,data)
            self.gplist.append(gp)
    
    def predict_mean(self,x):
        means = [gp.predict_batch(x)[0] for gp in self.gplist]
        means = np.hstack(means)
        mean = np.mean(means,axis=1)
        return mean



#==============================================================================
# OTHER THING
#==============================================================================
sigma = np.array([1.0,2.0,2.0])
mu = np.array([0.0,0.0,-3.0])
def prior_sampler(N):
    return sigma*(np.random.normal(size=(N,3)) + mu)

def prior_logpdf(sample):
    return np.sum(-(sample - mu)**2/(2*sigma**2) - sigma*np.log(SQRT2PI))


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

gpsamples = hmc_samples(kernel,noisekernel,(X,Y),hparams,300,
                        0.1,5)
gpjoiner = GPJoiner(kernel,noisekernel,gpsamples[100:,:],(X,Y))
hparams = np.array([0.5,0.2,0.05])
gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
gp = gp.optimize(verbose=1,opt_choice="cg",positives_default="bound",
                 beta_update_fn="FR",line_search_fn="goldstein")
##
xpred = np.linspace(0,1).reshape(-1,1)
ypred,_ = gp.predict_batch(xpred)
ypredmrg = gpjoiner.predict_mean(xpred)
#
plt.plot(X,Y,'b*')
plt.plot(xpred,ypred,'g')
plt.plot(xpred,ypredmrg,'m')
plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')