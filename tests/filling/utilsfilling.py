# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as spla

def autocorrelation_to_angles(Xlist):
    Xarray = np.array(Xlist)
    corrmatrix = np.corrcoef(Xarray)
    return correlation_to_angles(corrmatrix)


def correlation_to_angles(corrmatrix):
    corrchol = spla.cholesky(corrmatrix)
    tarray = np.array([])
    for i in range(1,corrchol.shape[0]):
        c = corrchol[:i+1,i]
        tnew = spherical_parametrization(c)
        tarray = np.hstack([tarray,tnew])
    return tarray


def spherical_parametrization(points):
    if len(points) == 2:
        return np.array([np.arccos(points[0])])
    else:
        theta0 = np.arccos(points[0])
        s0 = np.sin(theta0)
        assert s0 != 0
        others = spherical_parametrization(points[1:]/np.sin(theta0))
        return np.hstack([np.array([theta0]),others])

def sp_to_corr(plist):
    def sp(plist):
        if len(plist) == 1:
            p = plist[0]
            return np.array([np.cos(p),np.sin(p)])
        else:
            p = plist[0]
            u0 = sp(plist[1:])
            return np.hstack([np.cos(p),np.sin(p)*u0])
    def triangular(i):
        return i*(i+1)//2
    U = np.identity(len(plist))
    for i in range(1,U.shape[0]):
        pl = plist[triangular(i-1):triangular(i)]
        U[:i+1,i] = sp(pl)
    return np.matmul(U.transpose(),U)