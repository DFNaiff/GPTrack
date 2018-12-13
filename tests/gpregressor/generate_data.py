# -*- coding: utf-8 -*-
import numpy as np


def generate_sine_data(i):
    X = np.linspace(0,1,21)
    Y = np.sin(2*np.pi*X) + 0.25*np.random.randn(*X.shape)
    np.savez("sine_data_%i"%i,X=X,Y=Y)
    plt.plot(X,Y,'o')
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    generate_sine_data(1)