#==============================================================================
# Basic GP Regression
#==============================================================================
import sys
sys.path.append("../..")

import numpy as np
import torch
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


data = np.load("../datasets/process2b.npz")

ndata = 100
nsplit = 50
T = data['T'][:ndata].astype(float)
X = data['X'][:ndata]
Y = data['Y'][:ndata]
Z = data['Z'][:ndata]

def split_train_test(T,Z,p):
    nprevious = 100
    ind_list = np.arange(nprevious,dtype=int)
    inds_training = np.random.choice(ind_list,size=int(nprevious*(1-p)),
                                     replace=False)
#    inds_test = [i for i in ind_list if i not in inds_training]
    xtrain = T[inds_training].reshape(-1,1)
    ytrain = Z[inds_training].reshape(-1,1)
    return xtrain,ytrain
xtrain,ytrain = split_train_test(T,Z,0.5)
kernel = kernels.Constant()*kernels.IsoMatern12(1)
noise_kernel = kernels.IIDNoiseKernel()
hparams = [20.0,1.0,1.0]
bounds = [[0.1,50.0],[0.1,20.0],[0.01,10.0]]
gp = gpobject.GPObject(kernel,noise_kernel,hparams,[xtrain,ytrain])
gp = gp.optimize(option="B",num_starts=1,bounds=bounds,verbose=2,
                 to_optimize="loo_error")

Zpred = gp.predict_batch(T.reshape(-1,1))
plt.plot(T,Z,'b')
plt.plot(T,Zpred[0],'r--')
plt.plot(xtrain,ytrain,'go')
