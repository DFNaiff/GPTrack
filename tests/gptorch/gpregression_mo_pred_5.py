import sys
sys.path.append("../..")
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


data = np.load("../datasets/process2b.npz")

ndata = 100
nsplit = 50
T = data['T'][:ndata]
X = data['X'][:ndata]
Y = data['Y'][:ndata]
Z = data['Z'][:ndata]


def set_data(T,Ylist,Z,p=0.5,nprevious=10):
    """
        T : Time index
        Ylist : list of itens to be used for prediction
        Zlist : item to be predicted
        nprevious : time to train full until prediction
        p : percentage of nprevious to hide
    """
    ndata = len(T)
    nout = len(Ylist) + 1
    npred = ndata - nprevious
    ind_list = np.arange(nprevious,dtype=int)
    inds_training = np.random.choice(ind_list,size=int(nprevious*(1-p)),
                                     replace=False)
    ntraining = len(inds_training)
#    inds_test = [i for i in ind_list if i not in inds_training]
    marker_list = list(range(len(Ylist) + 1))
    xtrain = T[inds_training]
    ytrains = [Y[inds_training] for Y in Ylist]
    ztrain = Z[inds_training]
    prep1a = np.vstack([m*np.ones((ntraining,1)) for m in marker_list])
    prep1b = np.vstack([m*np.ones((npred,1)) for m in marker_list[:-1]])
    prep2a = np.tile(xtrain.reshape(-1,1),[nout,1])
    prep2b = np.tile(T[nprevious:].reshape(-1,1),[nout-1,1])
    Xtrain = np.hstack([prep1a,prep2a])
    Xdata = np.hstack([np.vstack([prep1a,prep1b]),
                       np.vstack([prep2a,prep2b])])
    print(Xdata)
    prepya = [ytrain.reshape(-1,1) for ytrain in ytrains] + \
             [ztrain.reshape(-1,1)]
    prepyb = [Y[nprevious:].reshape(-1,1) for Y in Ylist]
    Ytrain = np.vstack(prepya)
    Ydata = np.vstack(prepya + prepyb)
#    xtest = T[inds_test]
#    ytest = T[inds_test]
    return Xtrain,Ytrain,Xdata,Ydata
    
Xtrain,Ytrain,Xdata,Ydata = set_data(T,[X,Y],Z,nprevious = nsplit)
kernel = kernels.TensorProd(kernels.SphericalCorr(3),
                            kernels.IsoMatern32(dim=1))
noisekernel = kernels.MONoiseKernel(3)
#hparams = [1.0,1.0,np.pi/3,10.0,1e-2,1e-2]
positives = [True,True,False,True,True,True]
hparams = [0.7965539693832397, 0.8695805072784424, 0.8,
           1.0,1.0,0.8400896787643433, 60.11109161376953,
           1e-1,1e-1,1e-1]
#hparams = [0.8772075772285461, 1.386620283126831, 0.9989368319511414, 1.1128194332122803, 1.0215511322021484, 0.831591784954071, 59.16266632080078, 0.0501825176179409]
positives = [True,True,True,
             False,False,False,True,
             True,True,True]
bounds = [(1e-4,1e4)]*3+[(-np.pi,np.pi)]*3 + [(1e-4,1e4)] + \
         [(1e-8,1e3)]*3
#Kernel testing
#xkern,ykern = torch.tensor(xtrain)
gp = gpobject.GPObject(kernel,noisekernel,hparams,(Xtrain,Ytrain))

start_time = time.clock()
gp = gp.optimize(option="B",verbose=0,num_starts = 1,
                 bounds=bounds)
end_time = time.clock()
print(end_time - start_time)
hparams = gp.showhparams()
print(hparams)
gp = gpobject.GPObject(kernel,noisekernel,hparams,(Xdata,Ydata))

T0 = np.hstack([0.0*np.ones((ndata,1)),T.reshape(-1,1).astype(float)])
T1 = np.hstack([1.0*np.ones((ndata,1)),T.reshape(-1,1).astype(float)])
T2 = np.hstack([2.0*np.ones((ndata,1)),T.reshape(-1,1).astype(float)])
Xpred = gp.predict_batch(T0,getvar = False).flatten()
Ypred = gp.predict_batch(T1,getvar = False).flatten()
Zpred = gp.predict_batch(T2,getvar = False).flatten()
#Prediction
#Ypred = []
#Zpred = []
#for t in T:
#    ypred = gp.predict([[0.0,np.float(t)]])[0][0,0]
#    zpred = gp.predict([[1.0,np.float(t)]])[0][0,0]
#    Ypred.append(ypred)
#    Zpred.append(zpred)
plt.figure()
plt.plot(T,X,'b')
#plt.plot(T,xtrain[:ntraining,1],'go')
plt.plot(T,Xpred,'r')

plt.figure()
plt.plot(T,Y,'b')
#plt.plot(T,xtrain[:ntraining,1],'go')
plt.plot(T,Ypred,'r')
plt.figure()
plt.plot(T,Z,'b')
plt.plot(T,Zpred,'r')
plt.axvline(x=nsplit)