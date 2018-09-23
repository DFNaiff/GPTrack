import sys
sys.path.append("../..")

import numpy as np
import torch
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


data = np.load("../datasets/process2b.npz")

T = data['T'][:150]
X = data['X'][:150]
Y = data['Y'][:150]
Z = data['Z'][:150]


def set_data(T,Y,Z,p=0.7):
    ndata = len(T)
    nprevious = 50
    npred = ndata - nprevious
    ind_list = np.arange(nprevious,dtype=int)
    inds_training = np.random.choice(ind_list,size=int(nprevious*(1-p)),
                                     replace=False)
    ntraining = len(inds_training)
#    inds_test = [i for i in ind_list if i not in inds_training]
    xtrain = T[inds_training]
    ytrain = Y[inds_training]
    ztrain = Z[inds_training]
    prep1 = np.vstack([np.zeros((ntraining,1)),
                              np.ones((ntraining,1)),
                              np.zeros((npred,1))])
    prep2 = np.vstack([np.tile(xtrain.reshape(-1,1),[2,1]),
                       T[nprevious:].reshape(-1,1)])
    X = np.hstack([prep1,prep2])
    Y = np.vstack([ytrain.reshape(-1,1),ztrain.reshape(-1,1),
                   Y[nprevious:].reshape(-1,1)])
#    xtest = T[inds_test]
#    ytest = T[inds_test]
    return X,Y,ntraining
    
xtrain,ytrain,ntraining = set_data(T,Y,Z)
kernel = kernels.TensorProd(kernels.SphericalCorr(2),
                            kernels.IsoMatern12(dim=1))
noisekernel = kernels.IIDNoiseKernel()
hparams = [1.0,1.0,np.pi/3,10.0,1e-2]
positives = [True,True,False,True,True]
#Kernel testing
#xkern,ykern = torch.tensor(xtrain)
gp = gpobject.GPObject(kernel,noisekernel,hparams,(xtrain,ytrain))
gp = gp.optimize(positives)
print([p.item() for p in gp.hparams])
#Prediction
Ypred = []
Zpred = []
for t in T:
    ypred = gp.predict([[0.0,np.float(t)]])[0].numpy()[0,0]
    zpred = gp.predict([[1.0,np.float(t)]])[0].numpy()[0,0]
    Ypred.append(ypred)
    Zpred.append(zpred)
plt.figure()
plt.plot(T,Y,'b')
#plt.plot(T,xtrain[:ntraining,1],'go')
plt.plot(T,Ypred,'r')
plt.figure()
plt.plot(T,Z,'b')
#plt.plot(T,xtrain[ntraining:,1],'go')
plt.plot(T,Zpred,'r')