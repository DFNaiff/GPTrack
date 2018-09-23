import sys
sys.path.append("../..")

import numpy as np
import torch
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


data = np.load("../datasets/process1a.npz")

T = data['T']
Y = data['Y']
Z = data['Z']


def set_data(T,Y,Z,p=0.5):
    ind_list = np.arange(100,dtype=int)
    inds_training = np.random.choice(ind_list,size=int(100*(1-p)),
                                     replace=False)
    ntraining = len(inds_training)
#    inds_test = [i for i in ind_list if i not in inds_training]
    xtrain = T[inds_training]
    ytrain = Y[inds_training]
    ztrain = Z[inds_training]
    X = np.hstack([np.vstack([np.zeros((ntraining,1)),np.ones((ntraining,1))]),
                   np.tile(xtrain.reshape(-1,1),[2,1])])
    Y = np.vstack([ytrain.reshape(-1,1),ztrain.reshape(-1,1)])
#    xtest = T[inds_test]
#    ytest = T[inds_test]
    return X,Y,ntraining
    
xtrain,ytrain,ntraining = set_data(T,Y,Z)
kernel = kernels.TensorProd(kernels.SphericalCorr(2),
                            kernels.IsoRBF(dim=1))
noisekernel = kernels.IIDNoiseKernel()
hparams = [1.0,1.0,np.pi/3,1.0,1e-2]
positives = [True,True,False,True,True]
#Kernel testing
#xkern,ykern = torch.tensor(xtrain)
gp = gpobject.GPObject(kernel,noisekernel,hparams,(xtrain,ytrain))
gp = gp.optimize(positives)

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