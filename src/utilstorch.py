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


def binary_function_matrix_ret(f,x1,x2):
    """
        f : two argument kernel function
        x1 : array with n rows
        x2 : array with m rows
        returns : n x m tensor M, where M[i,j] = f(l_i,l_j)
    """
    n = x1.shape[0]
    m = x2.shape[0]
    X1 = torch.tensor(np.repeat(x1.numpy(),m,axis=0)).float()
    X2 = torch.tensor(np.tile(x2.numpy(),[n,1])).float()
    K = f(X1,X2)
    K = K.reshape(n,m)
    return K


def hypersphere_param(n,thetas):
    #Here, n is the dimension of the underlying space (for instance, 
    #n = 2 is the parametrisation of the circle, 3 parametrization sphere)
    w0 = [None]*n
    for i in range(n-1):
        w0i = torch.cos(thetas[i])
        for j in range(0,i):
            w0i = w0i*torch.sin(thetas[j])
        w0i = w0i.unsqueeze(0)
        w0[i] = w0i
    #Last one
    w0i = torch.sin(thetas[0])
    for j in range(1,n-1):
        w0i = w0i*torch.sin(thetas[j])
    w0i = w0i.unsqueeze(0)
    w0[-1] = w0i
    #Join everything
    w0 = torch.cat(w0)
    return w0


def relation_array(f,x,X):
    #TODO : Find a way to not pass through numpy
    X1 = torch.tensor(x.numpy()*np.ones(X.shape)).float()
    X2 = torch.tensor(X).float()
    k = f(X1,X2)
    return k