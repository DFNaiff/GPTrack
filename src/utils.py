# -*- coding: utf-8 -*-
import itertools

import numpy as np


def binary_function_matrix(f,L):
    """
        f : two argument function
        L : [l_1,...,l_n] list of arguments
        
        returns : n x n array M, where M[i,j] = f(l_i,l_j) 
    """
    M = np.array([[f(li,lj) for lj in L] for li in L])
    return M


def binary_function_matrix_vec(f,x):
    """
        f : two argument function
        x : an n x m array        
        returns : n x n array M, where M[i,j] = f(l_i,l_j) 
    """
    n = x.shape[0]
    X1 = np.repeat(x.numpy(),n,axis=0)
    X2 = np.tile(x.numpy(),[n,1])
    K = f(X1,X2)
    K = K.reshape(n,n)
    return K


def binary_function_matrix_2(f,L1,L2):
    """
        f : two argument function
        L1 : [l1_1,...,l1_m] list of arguments
        L2 = [l2_1,...,l2_n] list of arguments
        returns : m x n array M, where M[i,j] = f(l1_i,l2_j) 
    """
    M = np.array([[f(li,lj) for lj in L2] for li in L1])
    return M


def complete_with_zeros(x,n):
    """
        Complete 1d array with size m <= n with zeros,
        returning a 1d array with size n
    """
    return np.append(x,np.zeros(n-len(x)))


def equals_before(l,i):
    """
        Return the number of items in l before i equals to l[i],
    """
    return l[:i].count(l[i]) 


def invf(f):
    if f == np.log:
        return np.exp
    if f == np.tan:
        return np.arctan


def hypersphere_param(n,thetas):
    """
        (n-1)-sphere parametrization by 
        thetas = [theta_1,theta_2,...,theta_n],
        returning
        w = [cos(theta_1),sin(theta1)cos(theta2),...,
             sin(theta_1)...sin(theta_(n-1))cos(theta_n),
             sin(theta_1)...sin(theta_(n-1))sin(theta_n)]
    """
    if n == 2:
        w0 = np.array([np.cos(thetas[0]),np.sin(thetas[0])])
        return w0
    else:
        w0 = np.hstack([np.cos(thetas[0]),np.sin(thetas[0])*np.ones(n-1)])
        w0[1:] = w0[1:]*hypersphere_param(n-1,thetas[1:])
        return w0

        
def hypersphere_param_derivative(n,thetas,j):
    """
        partial derivative by theta_j of the
        (n-1)-sphere parametrization by 
        thetas = [theta_1,theta_2,...,theta_n], where, being 
        w = [cos(theta_1),sin(theta1)cos(theta2),...,
             sin(theta_1)...sin(theta_(n-1))cos(theta_n),
             sin(theta_1)...sin(theta_(n-1))sin(theta_n)],
        returns dv/d(theta_j)
    """
    dw0 = np.zeros(n)
    for k in range(n):
        if k + 1 < j: # Case (a), w0_k = 0
            dw0[k] = 0.0
        if k + 1 == j: # Case (b), w0_k = -prod_(i=1,...,k)sin(theta_i)
            dw0[k] = -np.prod(np.sin(thetas[:k+1]))
        if k + 1 > j and k + 1 < n: # Case (c)
            dw0[k] = np.prod(np.sin(thetas[:j-1]))*\
                     np.prod(np.sin(thetas[j:k-1]))*\
                     np.cos(thetas[j-1])*np.cos(thetas[k])
        if k + 1 == n: # Case (d)
            dw0[k] = np.prod(np.sin(thetas[:j-1]))*\
                     np.prod(np.sin(thetas[j:]))*\
                     np.cos(thetas[j-1])
    return dw0


def hypersphere_param_derivative2(n,thetas,j):
    """
        Second partial derivative by theta_j of the
        (n-1)-sphere parametrization by 
        thetas = [theta_1,theta_2,...,theta_n], where, being 
        w = [cos(theta_1),sin(theta1)cos(theta2),...,
             sin(theta_1)...sin(theta_(n-1))cos(theta_n),
             sin(theta_1)...sin(theta_(n-1))sin(theta_n)],
        returns d2v/d(theta_j)2
    """
    dw0 = np.zeros(n)
    for k in range(n):
        if k + 1 < j: # Case (a), w0_k = 0
            dw0[k] = 0.0
        if k + 1 == j: # Case (b)
            dw0[k] = -np.prod(np.sin(thetas[:k]))*np.cos(thetas[k])
        if k + 1 > j and k + 1 < n: # Case (c). Same as (b)
            dw0[k] = -np.prod(np.sin(thetas[:k]))*np.cos(thetas[k])
        if k + 1 == n: # Case (d)
            dw0[k] = -np.prod(np.sin(thetas))
    return dw0


def nth_index(iterable, value, n):
    matches = (idx for idx, val in enumerate(iterable) if val == value)
    return next(itertools.islice(matches, n-1, n), None)
    

def product_list_access(p,i,j):
    """
        Return the indexes to access a sublist of the flattened list 
        L as if it is a ndarray of shape p, in which we 
        access the j-th index of the i-th dimension of the array
    """
    k1 = int(np.prod(p[i+1:]))
    k2 = int(p[i]*k1)
    k3 = int(np.prod(p[:i]))
    indexes = []
    for i in range(k3):
        init = i*k2 + j*k1
        end = init + k1
        indexes += range(init,end)
    return indexes


def relation_array(f,x,xs):
    """
        f : two argument function
        x : argument
        xs : [x_1,...,x_n] list of arguments
        
        returns : [f(x,x_1),...,f(x,x_n)] array
    """
    return np.array([f(x,xk) for xk in xs])


def roundodd(x):
    """
        Rounds x to the nearest odd
    """
    if x < 2:
        return 1
    else:
        return 2*int(np.ceil(x/2.0) - 1)
    
    
def sqexp(x,y,l,theta=1.0):
    return theta*np.exp(-0.5*np.sum((np.square(x-y)/(l**2)),
                                    axis=1,keepdims=True))


def triangular(i):
    return i*(i+1)//2
    

def where_in_triangular(i):
    n = 1
    while True:
        if i > triangular(n-1) and i <= triangular(n):
            break
        else:
            n = n + 1
    return n