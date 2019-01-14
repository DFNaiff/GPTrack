# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


data = np.load("../datasets/process3a.npz")

ndata = 1000
ntraining = 50
Xtrue = 100*data['T'][:ndata]
Ytrue = data['X'][:ndata,0] - 3.0
samples = np.random.choice(np.arange(ndata,dtype=int),
                           size=ntraining,
                           replace=False)
X = Xtrue[samples].reshape(-1,1)
Y = Ytrue[samples].reshape(-1,1)
kernel = kernels.Constant()*kernels.IsoMatern52(dim=1)
noisekernel = kernels.IIDNoiseKernel()
hparams = np.array([1.0,10.0] + [1e-1])
bounds = [[0.01,10.0],[0.1,10.0]] + [[1e-3,1e1]]

#cvbatch = [list(range(i,i+1)) for i in range(0,21)]
gp = gpobject.GPObject(kernel,noisekernel,hparams,[X,Y])
gp,bic = gp.optimize(return_bic=True,bounds=bounds,verbose=2,opt_choice="cg",
                     beta_update_fn="FR",line_search_fn="goldstein")
print(bic)
xpred = np.linspace(0,1).reshape(-1,1)
ypred,cypred = gp.predict_batch(Xtrue.reshape(-1,1))

plt.plot(X,Y,'b*')
plt.plot(Xtrue,ypred,'b--')
plt.plot(Xtrue,Ytrue,'r--')