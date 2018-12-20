import sys
sys.path.append("../..")
import time

import numpy as np
import matplotlib.pyplot as plt

from src import kernels,gpobject

#This test is about filling an sde data, 
#with last point known. We use CholeskyCorr here

data = np.load("../datasets/process3a.npz")

ndata = 200
nsplit = 40
T = 100*data['T'][:ndata]
datay = data['X'][:ndata,:]
X,Y,Z = datay[:,0],datay[:,1],datay[:,2]
X = X - 3;Y = Y - 3; Z = Z - 3;


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
    marker_list = list(range(len(Ylist) + 1))
    xtrain = T[inds_training]
    ytrains = [Y[inds_training] for Y in Ylist]
    ztrain = Z[inds_training]
    prep1a = np.vstack([m*np.ones((ntraining,1)) for m in marker_list])
    prep1b = np.vstack([m*np.ones((npred,1)) for m in marker_list[:-1]])
    prep1b = np.vstack([prep1b,[[marker_list[-1]]]])
    prep2a = np.tile(xtrain.reshape(-1,1),[nout,1])
    prep2b = np.tile(T[nprevious:].reshape(-1,1),[nout-1,1])
    prep2b = np.vstack([prep2b,[[T[-1]]]])
    Xtrain = np.hstack([prep1a,prep2a])
    Xdata = np.hstack([np.vstack([prep1a,prep1b]),
                       np.vstack([prep2a,prep2b])])
    prepya = [ytrain.reshape(-1,1) for ytrain in ytrains] + \
             [ztrain.reshape(-1,1)]
    prepyb = [Y[nprevious:].reshape(-1,1) for Y in Ylist]
    prepyb += [[[Z[-1]]]]
    Ytrain = np.vstack(prepya)
    Ydata = np.vstack(prepya + prepyb)
    return Xtrain,Ytrain,Xdata,Ydata
    
Xtrain,Ytrain,Xdata,Ydata = set_data(T,[X,Y],Z,nprevious = nsplit)

#==============================================================================
# Make better initial guesses
#==============================================================================
rhoinit = 0.1
hinit = np.array([0.8,0.8,0.8])
corrmatrix = (1 - rhoinit)*np.identity(3) + rhoinit*np.ones((3,3))
covmatrix = np.outer(hinit,hinit)*corrmatrix
covchol = np.linalg.cholesky(covmatrix).transpose()
mokernelparams = list(np.hstack([np.diag(covchol,k=k) 
                        for k in range(2,-1,-1)]))
kernel = kernels.TensorProd(kernels.SphericalCorr(3),
                            kernels.IsoMatern12(dim=1))
noisekernel = kernels.MONoiseKernel(3)

positives = [True,True,False,True,True,True]
hparams = mokernelparams +\
           [60.11109161376953,
           1e-1,1e-1,1e-1]
positives = [False,False,False,
             True,True,True,
             True,
             True,True,True]
bounds = [(-100.0,100.0)]*3 + [(0.0,100.0)]*3 + [(1e-4,1e4)] + \
         [(1e-8,1e3)]*3
frozen = []
#Kernel testing
#xkern,ykern = torch.tensor(xtrain)
gp = gpobject.GPObject(kernel,noisekernel,hparams,(Xtrain,Ytrain))
print(gp.showhparams())
start_time = time.clock()
gp = gp.optimize(option="B",verbose=2,num_starts = 1,
                 bounds=bounds,line_search_fn = "goldstein")
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

def plot_train(ind,S,Y):
    s = [i for i in range(len(S[:,0])) if 
         int(S[i,0]) == ind]
    T = S[s,1]
    X = Y[s]
    plt.plot(T,X,'go')
    return X,T
plt.figure()
plt.plot(T,X,'b')
plt.plot(T,Xpred,'r')
plot_train(0,Xdata,Ydata)
plt.figure()
plt.plot(T,Y,'b')
plt.plot(T,Ypred,'r')
plot_train(1,Xdata,Ydata)
plt.figure()
plt.plot(T,Z,'b')
plt.plot(T,Zpred,'r')
plot_train(2,Xdata,Ydata)
plt.axvline(x=T[nsplit])