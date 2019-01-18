# -*- coding: utf-8 -*-

#Testing 1d warping

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


loadfile = np.load("sine_data_1.npz")
X = loadfile["X"].reshape(-1,1)
Y = loadfile["Y"].reshape(-1,1)
X = np.sqrt(X)

kernel = kernels.CompositionKernel(kernels.Constant()*kernels.IsoRBF(dim=1),
                                   kernels.KswamyWarping(1))
noisekernel = kernels.IIDNoiseKernel()
hparams = np.array([0.5,0.2,1.0,1.0,0.05])
#bounds = {2:[0.5,1.5],3:[0.5,1.5]}
bounds = {}
gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
gp = gp.optimize(bounds=bounds,verbose=2,opt_choice="lbfgs",positives_default="bound",
                 beta_update_fn="FR",line_search_fn="goldstein",
                 plb=1e-6)

xpred = np.linspace(0,1).reshape(-1,1)
#xpred = np.sqrt(xpred)
ypred,cypred = gp.predict_batch(xpred)

plt.plot(X,Y,'b*')
plt.plot(xpred,ypred,'b--')
plt.plot(xpred,np.sin(2*np.pi*xpred**2),'r--')