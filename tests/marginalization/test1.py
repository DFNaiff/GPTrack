# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import torch

from src import kernels,utilstorch,gpobject

SQRT2PI = float(np.sqrt(2*np.pi))


class NormPrior(object):
    """
        Dummy class for testing
    """
    def __init__(self):
        pass
    
    def sample(self,numsamples):
        return list(np.exp(np.random.normal(size=(numsamples,3))))
    
    def logpdf(self,samples):
        result = []
        for sample in samples:
            ss = torch.cat([s.unsqueeze(0) for s in sample])
            result.append(torch.prod(-torch.log(ss)**2/2 - \
                                     torch.log(ss*SQRT2PI)))
        result = np.array(result,dtype=object)
        return result

def smc_samples(kernel,noise_kernel,data,prior=None,
                numsamples = 3,P=10,K=5,
                **kwargs):
    numdata = data[0].shape[0]
    samples = prior.sample(numsamples)
    for i,sample in enumerate(samples): #Turn into tensor
        samples[i] = [torch.tensor(s,requires_grad=True) for s in sample]
    samples = np.array(samples,dtype=object)
    batchsize = numdata//P
    oldlogdensities = prior.logpdf(samples)
    oldpriors = prior.logpdf(samples)
    weights = 1/numsamples*np.ones(numsamples)
    splitted_data = split_data(data,batchsize)
    current_xdata = None
    current_ydata = None
    for data_batch in splitted_data:
        current_xdata,current_ydata = update_data(current_xdata,current_ydata,
                                                  data_batch)
        gplist = []
        for i,hparam in enumerate(samples):
            gplist.append(gpobject.GPObject(kernel,noise_kernel,hparam,
                                    (current_xdata,current_ydata)))
        logdensities = []
        for i,gp in enumerate(gplist):
            logdensities.append(gp.loglikelihood + oldpriors[i])
        logdensities = np.array(logdensities,dtype=object)
        print(logdensities)
        print(oldlogdensities)
        samples,weights = reweight_and_resample(samples,weights,
                                                oldlogdensities,
                                                logdensities)
        print(samples[0])
#        samples = mcmc_step(samples,logdensities,current_xdata,current_ydata,
#                            **kwargs)
        oldlogdensities = logdensities
        oldpriors = prior.logpdf(samples)
    gplist = [gpobject.GPObject(kernel,noise_kernel,hparam,
                                (current_xdata,current_ydata))
                    for hparam in samples]
    return gplist
    

#def mcmc_step(samples,logdensities,K,current_xdata,current_ydata,**kwargs):
#    epsilon = kwargs.get("epsilon",0.01)
#    for i,sample in enumerate(samples):
#        logdensity = logdensities[i]
#        logdensity.backward()
#        grads = np.array([s.grad for s in sample],dtype=object)
#        for k in range(K):
#            p = np.random.randn(len(sample))
#            hamiltonian = np.dot(p,p)/2 + logdensity
#            
#            p = p - epsilon*g/2
#            samplenew = sample + epsilon*p
#            for i,s in enumerate(samplenew):
#                samplenew[i] = s.clone().detach().requires_grad_()
#            gpnew = gpobject.GPObject(kernel,noise_kernel,samplenew,
#                                (current_xdata,current_ydata))
#            gpnew.loglikelihood + 
#    return samples

def reweight_and_resample(samples,weights,old_ld,ld):
    #TODO : Add the < N_e condition (find it in some paper
    old_ld = np.array([l.detach().numpy() for l in old_ld])
    ld = np.array([l.detach().numpy() for l in ld])
    weights = weights*np.exp(ld - old_ld)
    weights /= np.sum(weights)
    indsamples = np.random.choice(len(samples),size=len(samples),
                                  p=weights)
    samples = [samples[i] for i in indsamples] #TODO : Ugly workaround
    return samples,weights


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














#==============================================================================
# OTHER THING
#==============================================================================
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

print(smc_samples(kernel,noisekernel,(X,Y),NormPrior()))
#gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
#gp = gp.optimize(verbose=1,opt_choice="cg",positives_default="bound",
#                 beta_update_fn="FR",line_search_fn="goldstein")
#
#xpred = np.linspace(0,1).reshape(-1,1)
#ypred,cypred = gp.predict_batch(xpred)
#
#plt.plot(X,Y,'b*')
#plt.plot(xpred,ypred,'b--')
#plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')