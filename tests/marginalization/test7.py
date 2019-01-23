# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import torch
import emcee

from src import kernels,utilstorch,gpobject,margin




#==============================================================================
# OTHER THING
#==============================================================================

loadfile = np.load("sine_data_1.npz")
X = loadfile["X"].reshape(-1,1)
Y = loadfile["Y"].reshape(-1,1)

kernel = kernels.Constant()*kernels.IsoRBF(dim=1)
noisekernel = kernels.IIDNoiseKernel()
kernel.set_priors()
noisekernel.set_priors()
hparams = np.array([0.5,0.2,0.05])
bounds = {0:[1e-6,None],
          1:[1e-6,None],
          2:[1e-6,None]}
warpings = {0:'sqrt',1:'sqrt',2:'sqrt'}

gpsamples = margin.hmc_samples(kernel,noisekernel,(X,Y),hparams,100,
                        0.1,5,verbose=1)
gpjoiner = margin.MCGP(kernel,noisekernel,gpsamples[50:,:],(X,Y))
hparams = np.array([0.5,0.2,0.05])
gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
gp = gp.optimize(verbose=1,opt_choice="cg",positives_default="bound",
                 beta_update_fn="FR",line_search_fn="goldstein")
##
xpred = np.linspace(0,1).reshape(-1,1)
ypred,_ = gp.predict_batch(xpred)
ypredmrg,_ = gpjoiner.predict(xpred)
#
plt.plot(X,Y,'b*')
plt.plot(xpred,ypred,'g')
plt.plot(xpred,ypredmrg,'m')
plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')