# -*- coding: utf-8 -*-
import numpy as np
import torch


def binary_function_matrix(f,x):
    """
        f : two argument kernel function
        X : array
        returns : n x n tensor M, where M[i,j] = f(l_i,l_j)
    """
    #TODO : Find a way to not pass through numpy
    n = x.shape[0]
    X1 = torch.tensor(np.repeat(x.numpy(),n,axis=0)).float()
    X2 = torch.tensor(np.tile(x.numpy(),[n,1])).float()
    K = f(X1,X2)
    K = K.reshape(n,n)
    return K
    
def relation_array(f,x,X):
    #TODO : Find a way to not pass through numpy
    X1 = torch.tensor(x.numpy()*np.ones(X.shape)).float()
    X2 = torch.tensor(X).float()
    k = f(X1,X2)
    return k