import numpy as np
import matplotlib.pyplot as plt
import sdeint

A = np.array([[-0.5, -2.0],
              [ 2.0, -1.0]])
A = np.array([[-0.5,-2.0,-1.0],
              [2.0,-1.0,-0.5],
              [1.0,0.5,-0.5]])
B = np.diag([0.5, 0.5,0.5]) # diagonal, so independent driving Wiener processes
#B = np.array([[0.5,0.1],
#              [0.1,0.5]])
tspan = np.linspace(0.0, 10.0, 10001)
x0 = np.array([3.0, 3.0,3.0])

def f(x, t):
    return A.dot(np.sin(x))

def G(x, t):
    return B

result = sdeint.itoint(f, G, x0, tspan)
plt.plot(tspan,result[:,0],'b')
plt.plot(tspan,result[:,1],'r')
plt.plot(tspan,result[:,2],'g')