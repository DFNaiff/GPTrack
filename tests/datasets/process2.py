# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt

def main_process_step(x0,t,period,amplitude,skew,var):
    return x0 + var*spstats.skewnorm.rvs(-skew*x0) + amplitude*np.sin(period*t) 

def derivative_processes(y0,z0,x0,skew,var):
    y0 = y0 + var*spstats.skewnorm.rvs(-3*skew*x0)
    z0 = z0 + var*spstats.skewnorm.rvs(-3*skew*x0)
    return y0,z0
skew = 10.0
var = 0.1
period = 0.1
amplitude = 0.1
noise = 0.1
T = np.arange(201)
x0 = 0.0
y0 = 0.0
z0 = 0.0
X = [x0]
Y = [y0]
Z = [z0]
for t in T[1:]:
    x = main_process_step(x0,t,period,amplitude,skew,var)
    y,z = derivative_processes(y0,z0,x,skew,var)
    X.append(x + noise*np.random.randn())
    Y.append(y + noise*np.random.randn())
    Z.append(z + noise*np.random.randn())
    x0,y0,z0 = x,y,z
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
plt.figure()
plt.plot(T,X)
plt.figure()
plt.plot(T,Y,'g')
plt.plot(T,Z,'r')

np.savez("process2b.npz",T=T,X=X,Y=Y,Z=Z)