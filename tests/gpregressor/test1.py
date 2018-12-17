# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


loadfile = np.load("sine_data_1.npz")
X = loadfile["X"].reshape(-1,1)
Y = loadfile["Y"].reshape(-1,1)

kernel = kernels.Constant()*kernels.IsoRBF(dim=1)
noisekernel = kernels.IIDNoiseKernel()
hparams = np.array([1.0,1.0,1e-1])
bounds = [[0.01,10.0],[0.1,10.0],[1e-3,1e1]]
cvbatch = [list(range(6)),list(range(6,11)),
           list(range(11,16)),list(range(16,21))]
#cvbatch = [list(range(i,i+1)) for i in range(0,21)]
gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
gp = gp.optimize(bounds=bounds,verbose=2,
                 to_optimize = "crossvalidation",
                 cvbatch = cvbatch)

xpred = np.linspace(0,1).reshape(-1,1)
ypred,cypred = gp.predict_batch(xpred)

plt.plot(X,Y,'b*')
plt.plot(xpred,ypred,'b--')
plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')