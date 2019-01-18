# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from src import kernels,utilstorch,gpobject


loadfile = np.load("sine_data_2d_1.npz")
X = loadfile["X"].reshape(-1,1)
Y = loadfile["Y"].reshape(-1,1)
Z = loadfile["Z"].reshape(-1,1)

kernel = kernels.Constant()*kernels.IsoRBF(dim=2)
noisekernel = kernels.IIDNoiseKernel()
hparams = np.array([0.5,1.0,1.0])
warpings={0:'log',1:'log',2:'log'}
gp = gpobject.GPObject(kernel,noisekernel,hparams,[np.hstack([X,Y]),Z])
gp = gp.optimize(bounds={},verbose=2,opt_choice="lbfgs",
                 beta_update_fn="PR",line_search_fn="backtracking",
                 warpings=warpings)

import GPy
gpykern = GPy.kern.RBF(input_dim=2,variance=1.0,lengthscale=1.0,
                         ARD=False)
gpygp = GPy.models.GPRegression(np.hstack([X,Y]),Z,gpykern,
                                noise_var=1.0)

#xpred1,xpred2 = np.meshgrid(np.linspace(0,1,21),np.linspace(0,1,21))
#xpred = np.hstack([xpred1.reshape(-1,1),xpred2.reshape(-1,1)])
#ypred,cypred = gp.predict_batch(xpred)
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xpred1,xpred2,ypred.reshape(*xpred1.shape),color='red')
#ax.plot_surface(xpred1,xpred2,np.sin(2*np.pi*xpred1) + np.sin(2*np.pi*xpred2),color='blue')
#
#
#plt.plot(X,Y,'b*')
#plt.plot(xpred,ypred,'b--')
#plt.plot(xpred,np.sin(2*np.pi*xpred),'r--')