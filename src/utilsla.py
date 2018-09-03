# -*- coding: utf-8 -*-
import choldate
import numpy as np
import scipy.linalg as spla


def bilinear_form(u,A,w):
    return np.dot(u,np.dot(A,w))


def block(arrays):
    return np.vstack([np.hstack(L) for L in arrays])
    

def determinant_positive_definite(M):
    U = spla.lapack.dpotrf(M)[0]
    return (np.product(np.diag(U)))**2


def determinant_positive_definite_lower(L):
    return (np.product(np.diag(L)))**2


def contract_cholesky(U,i=0):
    """
        if i = 0:
            U : upper cholesky factor of a positive-definite matrix M
                of the form 
                    c v^T
                    v M
            returns : upper cholesky factor of M
    """
    if i == 0:
        v = U[0,1:]
        S = U[1:,1:]
        choldate.cholupdate(S,v)
        return S
    elif i == U.shape[0] - 1:
        return U[:-1,:-1]
    else:
        S11 = U[:i,:i]
        S13 = U[:i,i+1:]
        S31 = np.zeros_like(S13).transpose()
        S33 = U[i+1:,i+1:]
        v = U[i,i+1:]
        choldate.cholupdate(S33,v)
        return np.block([[S11,S13],
                         [S31,S33]])


def contract(K,i=0):
    """
        if i = 0:
            K : 2-d array of the form
                of the form 
                    c v^T
                    u M
            returns : M
        if i != 0:
            K : 2-d array of the form
                K1  b K2
                w^T c v^T
                K3  u K4
            returns : K1 K2
                      K3 K4
    """
    if i == 0:
        return K[1:,1:]
    else:
        return np.block([[K[:i,:i],K[:i,i+1:]],
                         [K[i+1:,:i],K[i+1:,i+1:]]])


def expand_cholesky(U,v,c):
    """
       U : upper cholesky factor of a positive-definite matrix K
       v : a n shaped 1d array
       c : a scalar
       
       Assumes that [K v^T, v c] will be positive-definite
       
       returns : upper cholesky factor of the matrix
           K v^T
           v c
    """
    S11 = U
    s21 = spla.solve_triangular(S11,v,lower=False,trans='T')
    s22 = np.sqrt(c - np.dot(s21,s21))
    return expand_triangular(S11,s21,s22)


def expand_cholesky_with_matrix(U,V,C):
    """
       U : upper cholesky factor of a positive-definite matrix K
       V : a (n,d) 2d aray
       C : a (d,d) 2d array
       
       Assumes that [K V^T, V C] will be positive-definite
       
       returns : upper cholesky factor of the matrix
           K   V
           V^T C
    """
    S11 = U
    S21 = spla.solve_triangular(S11,V,lower=False,trans='T')
    S22 = spla.cholesky(C - np.dot(S21.transpose(),S21))
    return expand_triangular_with_matrix(S11,S21,S22)


def expand_symmetric(K,v,c):
    """
        K : a (n,n) 2d array
        v : a n shaped 1d array
        c : a scalar
        
        returns: a (n + 1, n + 1) 2d array of the form
            K    v^T
            v    c
    """
    newK = np.hstack([K,v.reshape(-1,1)])
    newK = np.vstack([newK,np.append(v,c)])
    return newK


def expand_symmetric_with_matrix(K,V,C):
    """
       U : upper cholesky factor of a positive-definite matrix K
       V : a (n,d) 2d aray
       C : a (d,d) 2d array
        
        returns: a (n + d, n + d) 2d array of the form
            K      V
            V^T    C
    """
    newK = np.hstack([K,V])
    newK = np.vstack([newK,np.hstack([V.transpose(),C])])
    return newK
    

def expand_symmetric_and_invert(K,v,c):
    """
        K : a (n,n) 2d array
        v : a n shaped 1d array
        c : a scalar
        
        returns: a (n + 1, n + 1) 2d array, inverse of the matrix
            K    v^T
            v    c
    """
    newK = expand_symmetric(K,v,c)
    invnewK = inverse_cholesky(newK)
    return invnewK


def expand_symmetric_inverse_with_inverse(K,v,c,invK):
    """
        K : a (n,n) 2d array
        v : a n shaped 1d array
        c : a scalar
        invK : inverse of K
        
        returns: a (n + 1, n + 1) 2d array, inverse of the matrix
            K    v^T
            v    c
    """
    delta = c - bilinear_form(v,invK,v)
    w = -np.matmul(invK,v)/delta
    Q = invK + 1.0/delta*np.matmul(invK,np.matmul(np.outer(v,v),invK))
    invnewK = expand_symmetric(Q,w,delta)
    return invnewK


def expand_triangular(U,v,c):
    """
        U : a (n,n) 2d array upper triangular
        v : a n shaped 1d array
        c : a scalar
        
        returns: a (n + 1, n + 1) 2d array of the form
            K    v^T
            0    c
    """
    newU = np.hstack([U,v.reshape(-1,1)])
    newU = np.vstack([newU,np.append(np.zeros_like(v),c)])
    return newU
    

def expand_triangular_with_matrix(U,V,C):
    """
       U : upper cholesky factor of a positive-definite matrix K
       V : a (n,d) 2d aray
       C : a (d,d) 2d array
        
        returns: a (n + d, n + d) 2d array of the form
            K    V
            0    C
    """
    newU = np.hstack([U,V])
    newU = np.vstack([newU,
                      np.hstack([np.zeros_like(V).transpose(),C])])
    return newU
    

def inverse_cholesky(M):
    L = spla.cholesky(M,lower=True)    
    invL = spla.lapack.dpotri(L,lower=True)[0]
    invK = invL + (invL - np.diag(np.diag(invL))).transpose()
    return invK


def inverse_cholesky_lower(L):
    invL = spla.lapack.dpotri(L,lower=True)[0]
    invK = invL + (invL - np.diag(np.diag(invL))).transpose()
    return invK


def inverse_cholesky_upper(U):
    invU = spla.lapack.dpotri(U,lower=False)[0]
    invK = invU + (invU - np.diag(np.diag(invU))).transpose()
    return invK
    

def invlmatmul(U,v,trans='N'):
    return spla.solve_triangular(U,v,lower=True,trans=trans)


def invumatmul(U,v,trans='N'):
    return spla.solve_triangular(U,v,lower=False,trans=trans)