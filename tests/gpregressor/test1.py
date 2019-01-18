# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject

def smc_samples(kernel,noise_kernel,prior,data,P=10):
    N = len(data)
    batches = N//P
    samples = prior.sample(N)
    splitted_data = split_data(data,batches)
    for data_batch in splitted_data:
        gp = gpobject.GPObject()

#loadfile = np.load("sine_data_1.npz")
#X = loadfile["X"].reshape(-1,1)
#Y = loadfile["Y"].reshape(-1,1)
#
#kernel = kernels.Constant()*kernels.IsoRBF(dim=1)
#noisekernel = kernels.IIDNoiseKernel()
#hparams = np.array([0.5,0.2,0.05])
#bounds = {0:[1e-6,None],
#          1:[1e-6,None],
#          2:[1e-6,None]}
#warpings = {0:'sqrt',1:'sqrt',2:'sqrt'}
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