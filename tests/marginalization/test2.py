# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import torch

from src import kernels,utilstorch,gpobject

SQRT2PI = float(np.sqrt(2*np.pi))


def smc_samples(kernel,noise_kernel,data,
                prior_sampler,prior_logpdf,
                numsamples = 20,P=10,K=5,
                **kwargs):
    numdata = data[0].shape[0]
    samples = prior_sampler(numsamples) #Get samples from prior distribution
    for i,sample in enumerate(samples): #Turn samples into tensors
        samples[i] = [torch.tensor(s,requires_grad=True) for s in sample]
    samples = np.array(samples,dtype=object) #Array of tensors (IMPORTANT)
    batchsize = numdata//P #Batch size
    oldlogdensities = np.array([prior_logpdf(sample) for sample in samples],
                                dtype=object) #This is pi_0 for each samples
    oldpriors = np.array([prior_logpdf(sample) for sample in samples],
                          dtype=object) #This is the prior for each samples 
                                        #(Important, will turn into different
                                        # variables)
    weights = 1/numsamples*np.ones(numsamples)
    splitted_data = split_data(data,batchsize)
    current_xdata = None
    current_ydata = None
    for data_batch in splitted_data:
        current_xdata,current_ydata = update_data(current_xdata,current_ydata,
                                                  data_batch)
        gplist = []
        for i,hparam in enumerate(samples):
            print(hparam)
            gplist.append(gpobject.GPObject(kernel,noise_kernel,hparam,
                                    (current_xdata,current_ydata)))
        logdensities = []
        for i,gp in enumerate(gplist):
            logdensities.append(gp.loglikelihood + oldpriors[i])
        logdensities = np.array(logdensities,dtype=object)
        samples,weights,indsamples = reweight_and_resample(samples,weights,
                                                oldlogdensities,
                                                logdensities)
        logdensities = logdensities[indsamples]
        samples = mcmc_step(samples,K,
                             kernel,noise_kernel,
                             current_xdata,current_ydata,
                             prior_logpdf,
                             **kwargs)
        logdensities = logdensities[indsamples]
        oldpriors = np.array([prior_logpdf(sample) for sample in samples],
                              dtype=object)
    gplist = []
    for i,sample in enumerate(samples):
        hparams = [s.detach() for s in sample]
        gplist.append(gpobject.GPObject(kernel,noise_kernel,hparams,
                                (current_xdata,current_ydata)))
    return gplist
    

def mcmc_step(samples,K,
              kernel,noise_kernel,current_xdata,current_ydata,
              prior_logpdf,**kwargs):
    #For now recalculating one GP more than necessary.
    #Should result in a slowdown of 1/K percent.
    epsilon = kwargs.get("epsilon",0.01)
    for i,sample in enumerate(samples): #Do MCMC steps for each sample
        gp = gpobject.GPObject(kernel,noise_kernel,sample,
                                (current_xdata,current_ydata))
        logdensity = gp.loglikelihood + prior_logpdf(sample) 
        #Objective function for ith sample
        #Here, it has to be noticed that the density is e**u, not 
        #e**(-u). Hence, where logdensity and grads appears, the sign is
        #changed.
        #Get gradients
        grads = np.array(torch.autograd.grad(logdensity,sample,allow_unused=True),
                         dtype=object)
        print(i)
        for k in range(K):
            p = np.random.randn(len(sample)) #Momentum
            hamiltonian = np.dot(p,p)/2 - logdensity.detach().numpy()
            #Leapfrog step
            p = p + epsilon*grads/2
            samplenew = sample + epsilon*p
            print(samplenew)
            gpnew = gpobject.GPObject(kernel,noise_kernel,samplenew,
                                (current_xdata,current_ydata))
            logdensitynew = gpnew.loglikelihood + prior_logpdf(samplenew)
            gradsnew = np.array(torch.autograd.grad(logdensitynew,sample,
                                                    allow_unused=True),
                                dtype=object)
            p = p + epsilon*gradsnew/2
            
            hamiltoniannew = np.dot(p,p).numpy()/2 - \
                             logdensitynew.detach().numpy()
            dH = hamiltoniannew - hamiltonian
            accept = (dH < 0) or (np.random.random() < np.exp(-dH))
            if accept:
                grads = gradsnew.copy()
                logdensity = logdensitynew.clone()
                sample = samplenew.copy()
            else:
                continue
        samples[i] = sample #Maybe detach. But let's try it
    return samples

def reweight_and_resample(samples,weights,old_ld,ld):
    #TODO : Add the < N_e condition (find it in some paper
    old_ld = np.array([l.detach().numpy() for l in old_ld])
    ld = np.array([l.detach().numpy() for l in ld])
    weights = weights*np.exp(ld - old_ld)
    weights /= np.sum(weights)
    indsamples = np.random.choice(len(samples),size=len(samples),
                                  p=weights)
    samples = [samples[i] for i in indsamples] #TODO : Ugly workaround
    weights = 1.0/len(weights)*np.ones_like(weights)
    return samples,weights,indsamples


def split_data(data, group_size):
    for i in range(0, len(data[1]), group_size):
        yield (data[0][i:i+group_size,:],data[1][i:i+group_size,:])


def update_data(xdata,ydata,data_batch):
    if type(xdata) == type(None):
        return data_batch
    else:
        xdata = np.vstack([xdata,data_batch[0]])
        ydata = np.vstack([ydata,data_batch[1]])
        return xdata,ydata

class GPJoiner(object):
    def __init__(self,gplist):
        self.gplist = gplist
    
    def predict_mean(self,x):
        means = [gp.predict_batch(x)[0] for gp in self.gplist]
        means = np.hstack(means)
        mean = np.mean(means,axis=1)
        return mean



#==============================================================================
# OTHER THING
#==============================================================================
sigma = 3.0
def prior_sampler(N):
    return list(np.exp(sigma*np.random.normal(size=(N,3))))

def prior_logpdf(sample):
    ss = torch.cat([s.unsqueeze(0) for s in sample])
    return torch.sum(-torch.log(ss)**2/(2*sigma**2) - torch.log(ss*sigma*SQRT2PI))


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

gpsamples = smc_samples(kernel,noisekernel,(X,Y),prior_sampler,prior_logpdf)
gpjoiner = GPJoiner(gpsamples)
gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
gp = gp.optimize(verbose=1,opt_choice="cg",positives_default="bound",
                 beta_update_fn="FR",line_search_fn="goldstein")
#
xpred = np.linspace(0,1).reshape(-1,1)
ypred,_ = gp.predict_batch(xpred)
ypredmrg = gpjoiner.predict_mean(xpred)
#
plt.plot(X,Y,'b*')
plt.plot(xpred,ypred,'g')
plt.plot(xpred,ypredmrg,'m')
plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')