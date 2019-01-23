# -*- coding: utf-8 -*-
import numpy as np
from ..gpobject import GPObject


#TODO : Too much memory and computational consumption
#Needs to consider redundant samples from MC (since we are mainly 
#doing MCMC, no need for holding two gps for the same sample)

class MCGP(object):
    def __init__(self,kernel,noise_kernel,gpsamples,data):
        self.gplist = []
        for i,sample in enumerate(gpsamples):
            gp = GPObject(kernel,noise_kernel,sample,data)
            self.gplist.append(gp)
    
    def predict(self,x):
        """
            Get mean and variance. Remember, since 
            the model is now a mixture of gaussians, 
            variance may not be a representative quantity 
            for uncertainty
        """
        means,variances = list(zip(*[gp.predict_batch(x) for gp in self.gplist]))
        means = np.hstack(means)
        variances = np.hstack(variances)
        mean = np.mean(means,axis=1)
        var = np.mean(means**2 + variances,axis=1) - mean**2
        return mean,var