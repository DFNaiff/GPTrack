# -*- coding: utf-8 -*-
import numpy as np


def generate_sine_data(i):
    X = np.linspace(0,1,21)
    Y = np.sin(2*np.pi*X) + 0.25*np.random.randn(*X.shape)
    np.savez("sine_data_%i"%i,X=X,Y=Y)
    plt.plot(X,Y,'o')

def generate_square_data(i):
    XY = np.random.uniform((0.0,0.0),(1.0,1.0),size=(51,2))
    X = XY[:,0]
    Y = XY[:,1]
    Z = np.sin(2*np.pi*X) + np.sin(2*np.pi*Y) + 0.25*np.random.randn(*X.shape)
    np.savez("sine_data_2d_%i"%i,X=X,Y=Y,Z=Z)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    generate_square_data(1)