# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import torch
import emcee

from src import kernels,utilstorch,gpobject

SQRT2PI = float(np.sqrt(2*np.pi))


def smc_samples(kernel,noise_kernel,data,
                prior_sampler,prior_logpdf,
                numsamples = 20,P=10,K=20,
                **kwargs):
    def logpdf(sample,data):
        posinds = kernel.positives + noisekernel.positives
        plnpdf = prior_logpdf(sample)
        sample = sample.copy()
        sample[posinds] = np.exp(sample[posinds])
        hparams = [torch.tensor(s) for s in sample]
        try:
            gp = gpobject.GPObject(kernel,noisekernel,hparams,data)
        except:
            res = -100000.0
            print(res,'error')
            return res
        res = gp.loglikelihood.numpy()
        res += plnpdf
        print(res)
        return res
    #If variable is positive, prior and pdf from samples
    #MUST COME FROM LOGARITHM
    numdata = data[0].shape[0]
    ndim = kernel.nhyper + noisekernel.nhyper
    samples = prior_sampler(numsamples) #Get samples from prior distribution
    batchsize = numdata//P #Batch size
    oldlogdensities = np.array([prior_logpdf(sample) for sample in samples]) #pi_0
    weights = 1/numsamples*np.ones(numsamples) #weights
    splitted_data = split_data(data,batchsize) #Iterator for splitted data
    current_xdata = None
    current_ydata = None
    for i,data_batch in enumerate(splitted_data):
        current_xdata,current_ydata = update_data(current_xdata,current_ydata,
                                                  data_batch) #Add data
        if i == 0:
            emcee_sampler = emcee.EnsembleSampler(numsamples, ndim, logpdf,
                                                  args=[(current_xdata,current_ydata)])
        logdensities = np.array([logpdf(sample,data) for sample in samples]) #New log densities
        samples,weights,indsamples = reweight_and_resample(samples,weights,
                                                           oldlogdensities,
                                                           logdensities) #Resampling
        samples,_,_ = emcee_sampler.run_mcmc(samples, K)
        oldlogdensities = logdensities[indsamples]
        print(samples)
#        gplist = [] #List of gps
#        for i,sample in enumerate(samples):
#            hparam = sample.copy()
#            hparams[posinds] = np.exp(hparams[posinds]) #Deal if positives var
#            gplist.append(gpobject.GPObject(kernel,noise_kernel,hparam,
#                                    (current_xdata,current_ydata)))
#        logdensities = []
#        for i,gp in enumerate(gplist):
#            logdensities.append(gp.loglikelihood + oldpriors[i])
#        logdensities = np.array(logdensities)
#        samples,weights,indsamples = reweight_and_resample(samples,weights,
#                                                oldlogdensities,
#                                                logdensities)
#        logdensities = logdensities[indsamples]
#        samples = mcmc_step()
#        logdensities = logdensities[indsamples]
#        oldpriors = np.array([prior_logpdf(sample) for sample in samples])
#    gplist = []
#    for i,sample in enumerate(samples):
#        gplist.append(gpobject.GPObject(kernel,noise_kernel,sample,
#                                (current_xdata,current_ydata)))
#    return gplist


def reweight_and_resample(samples,weights,old_ld,ld):
    #TODO : Add the < N_e condition (find it in some paper
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

gpsamples = smc_samples(kernel,noisekernel,(X,Y),prior_sampler,prior_logpdf)
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