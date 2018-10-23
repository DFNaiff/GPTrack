# -*- coding: utf-8 -*-
import copy
import functools
import itertools

import numpy as np
import scipy.stats as spstats

from . import gpobject
from . import kernels
from . import utils
from . import utilsla

#
# GP with marginalization according to
# "Real-Time Information Processing of
# Environmental Sensor Network Data using
# Bayesian Gaussian Processes", M. A. Osborne, S. J. Roberts
#

#TODO : Renaming. phi_anything should be hparams_anything
class GPMarginGrid(object):
    def __init__(self,kernel,phisamples,positives,data,
                      **kwargs):
        """
            kernel : kernel of the GPs
            phisamples : if weights_term_construction = "grid":
                             list of samples for each hyperparameter dimension, 
                             [[phi_1^(1),...,phi_1^(m_1)],...,
                               [phi_n^(1),...,phi_n^(m_n)]]
                              where n = kernel.nhyper + noisekernel.nhyper
                              if weights_term_construction = "nongrid":
                              [phi_1,...,phi_n],
                              with phi_i = [phi_i^(1),...,phi_i^(n)]
                              where n = kernel.nhyper + noisekernel.nhyper
            positives : [bool]*n list of positives hyperparameters
            data : data supplied to GP. If supplied, has to be of the form
                   [xdata,zdata], where xdata,zdata are two lists of data.
                   If None, kernel is considered to not be initialized.
            noisekernel : kernel for the noise. If "DEFAULT", 
                          IIDKernel is chosen
            weights_term_construction : "grid" or "nongrid", depending on how to 
                                        construct. Default: "grid"
            prior_variances : a (positive) scalar or array
            gp_length_scales : a (positive) scalar or "ESTIMATE"
            verbose : degree of verbosity
        """
        self.kernel = kernel
        noisekernel = kwargs.get("noisekernel","DEFAULT")
        if noisekernel == "DEFAULT":
            self.noisekernel = kernels.IIDNoiseKernel()
        else:
            self.noisekernel = noisekernel
        self.phisamples = phisamples
        self.positives = positives
        self.weights_term_construction = kwargs.get("weights_term_construction",
                                                    "grid")
        self.prior_variances = kwargs.get("prior_variances",1.0)
        self.gp_length_scales = kwargs.get("gp_length_scales",1.0)
        self.verbose = kwargs.get("verbose",0)
        if data:
            self.initialize(data,self.weights_term_construction)
        else:
            self._initialized = False
            
    def predict(self,xs):
        """
            xs = [x_1,...,x_m] list of desired predictions
            returns : 
                means = [mean_1,...,mean_m] list of predictions means
                covs = [cov_1,...,cov_m] list of predicitions vars
                
        """
        self._check_initialized()
        if type(xs) not in [list,tuple]:
            return self._predict_single(xs)
        else:
            return self._predict_batch(xs)
    
    def initialize(self,data,weights_term_construction):
        #Total number of samples
        if weights_term_construction == "grid": #Grid parameters were supplied
            self.eta = np.product([len(phisample) 
                                   for phisample in self.phisamples])
            self.prior_means = _get_prior_means(self.phisamples,
                                                self.positives)
        elif weights_term_construction == "nongrid":
            self.eta = len(self.phisamples)
            phigrid = list(np.array(self.phisamples).transpose())
            self.prior_means = _get_prior_means(phigrid,
                                                self.positives)
        else:
            raise TypeError
        self.xdata = copy.copy(data[0])
        self.zdata = copy.copy(data[1])
        if self.verbose >= 1:
            print("Setting GPS")
        self._set_gps([self.xdata,self.zdata],weights_term_construction)
        # Calculate the weights
        if self.verbose >= 1:
            print("Calculating weight terms")
        if weights_term_construction == "grid":
            self.weights_term = _calculate_weights_term_grid(self.phisamples,
                                                             self.prior_variances,
                                                             self.gp_length_scales,
                                                             self.positives,
                                                             self.prior_means)
        elif weights_term_construction == "nongrid":
            self.weights_term = _calculate_weights_term_nongrid(self.phisamples,
                                                                self.prior_variances,
                                                                self.gp_length_scales,
                                                                self.positives,
                                                                self.prior_means)

        self._determine_weights_vector()
        if self.verbose >= 1 : 
            print("Initialized")
        self._initialized = True
    
    def _predict_single(self,x):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict(x)
            m_list[i] = m_i
            C_list[i] = C_i
        m,C = _combine_predictions_single(self.rho_vector,m_list,C_list)
        return m,C
    
    def _predict_batch(self,xs,retdiag = True):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict_batch(xs,retdiag=False)
            m_list[i] = m_i
            C_list[i] = C_i
        m,C = _combine_predictions_batch(self.rho_vector,m_list,C_list)
        if not retdiag:
            return m,C
        else:
            return m,np.diag(C).reshape(-1,1)
    
    def _check_initialized(self):
        if not self._initialized:
            raise TypeError #TODO : Change error
    
    #Initialization parameters
    def _set_gps(self,data,weights_term_construction):
        self.gplist = [None]*self.eta
        if weights_term_construction == "grid":
            phiiter = itertools.product(*self.phisamples)
        elif weights_term_construction == "nongrid":
            phiiter = self.phisamples
        for i,phisample in enumerate(phiiter):
            phisample = np.array(phisample)
            # Supply ith gaussian process with the mean function meanfunc 
            # and covariance function cov func
            self.gplist[i] = gpobject.GPObject(self.kernel,self.noisekernel,
                                               phisample,data)
    
    def _determine_weights_vector(self):
        self.rho_vector = _determine_weights(self.weights_term,self.gplist)
#==============================================================================
# AUXILIARY FUNCTIONS
#==============================================================================
def _calculate_weights_term_nongrid(phisamples,prior_variances,gp_length_scales,
                                    positives,prior_means=0.0):
    #TODO : Check whether Osborne's weight calculation is correct.
    #       To elaborate: The integral you've calculated should
    #       be the same as his. But maybe he's wrong (or it should 
    #       not be the same)
    """
        Calculates the weight term matrix without grid 
        phisamples : [phi_1,...,phi_eta] list of hyperparameters samples,
                     where phi_i = [phi_i^1,...,phi_i^n]
        prior_variances : a (positive) scalar (LATER: also arrays)
        gp_length_scales : a (positive) scalar (LATER: also arrays)
        positives : [bool]*n list of parameters that are positive
        prior_means : a scalar or n-sized array
        returns a eta sized square matrix
    """
    eta = len(phisamples)
    n = len(phisamples[0])
    phisamplestr = [None]*eta
    for i,phisample in enumerate(phisamples):
        # Apply log to positive hyperparameters
        phisampletr = phisample.copy()
        phisampletr[positives] = np.log(phisample[positives])
        phisamplestr[i] = phisampletr
    if type(gp_length_scales) != np.ndarray: # A scalar was supplied
        gp_length_scales = gp_length_scales*np.ones(n)
    if type(prior_variances) != np.ndarray: # A scalar was supplied
        prior_variances = prior_variances*np.ones(n)
    if type(prior_means) != np.ndarray: # A scalar was supplied
        prior_means = prior_means*np.ones(n)
    phisamplestr = np.array(phisamplestr)
    # Multivariate normal used for W
    # TODO: change for more efficient manner
    l2 = gp_length_scales**2
    C = utilsla.block([[np.diag(prior_variances + l2),np.diag(prior_variances)],
                       [np.diag(prior_variances),np.diag(prior_variances + l2)]])
    mvn = spstats.multivariate_normal(mean = np.hstack([prior_means]*2),
                                      cov=C)
    wfunc = lambda phi_i,phi_j : mvn.pdf(np.hstack([phi_i,phi_j])).reshape(-1,1)
    kfunc = functools.partial(utils.sqexp,l=gp_length_scales)
    K = utils.binary_function_matrix_vec(kfunc,phisamplestr)
    U = utilsla.spla.cholesky(K,lower=False)
    W = utils.binary_function_matrix_vec(wfunc,phisamplestr)
    return W,U


def _calculate_weights_term_grid(phisamples,prior_variances,gp_length_scales,
                           positives,prior_means=0.0):
    """
        Calculates the weight term matrix (actually, returns 
        the necessary factors)
        phisamples : list of samples for each hyperparameter dimension, 
                     [[phi_1^(1),...,phi_1^(m_1)],...,
                      [phi_n^(1),...,phi_n^(m_n)]]
        prior_variances : a (positive) scalar or n-sized array
        gp_length_scales : a (positive) scalar or n-sized array
        positives : [bool]*n list of positives hyperparameters
        prior_means : a scalar or n-sized array
        returns two m_1*...*m_n sized square matrix,
        calN and upper cholesky factor of K
    """
    n = len(phisamples)
    trphisamples = copy.deepcopy(phisamples)
    for i,_ in enumerate(phisamples):
        if positives[i]:
            trphisamples[i] = np.log(phisamples[i])
    if type(prior_variances) != np.ndarray: # A scalar was supplied
        prior_variances = prior_variances*np.ones(n)
    if type(prior_means) != np.ndarray: # A scalar was supplied
        prior_means = prior_means*np.ones(n)
    if type(gp_length_scales) != np.ndarray: # A scalar was supplied
        gp_length_scales = gp_length_scales*np.ones(n)
    #Make multivariate gaussians
    mvns = []
    for i in range(n):
        lambd = prior_variances[i]
        nu = prior_means[i]
        w = gp_length_scales[i]**2
        m = np.array([nu,nu])
        C = np.array([[lambd + w, lambd],
                      [lambd, lambd + w]])
        mvn = spstats.multivariate_normal(mean=m,cov=C)
        mvns.append(mvn)
    def caln(phi_i,phi_j,i):
        return mvns[i].pdf(np.hstack([phi_i,phi_j]))
    # Calculate Kronecker products
    # First matrices
    calni = functools.partial(caln,i=0)
    kerneli = functools.partial(utils.sqexp,
                                l = gp_length_scales[0])
    A = utils.binary_function_matrix(calni,trphisamples[0])
    K = utils.binary_function_matrix(kerneli,trphisamples[0])
    B = utilsla.spla.cholesky(K,lower=False)
    W = A.copy()
    U = B.copy()
    # Other matrices
    for i,_ in enumerate(trphisamples[1:],1):
        calni = functools.partial(caln,i=i)
        kerneli = functools.partial(utils.sqexp,
                                    l = gp_length_scales[i])
        A = utils.binary_function_matrix(calni,trphisamples[i])
        K = utils.binary_function_matrix(kerneli,trphisamples[i])
        B = utilsla.spla.cholesky(K,lower=False)
        W = np.kron(W,A)
        U = np.kron(U,B)
    return W,U


def _combine_predictions_single(rho_vector,m_list,C_list):
    #TODO : May be wrong
    marray = np.array(m_list)
    Carray = np.array(C_list)
    m = np.average(marray,weights=rho_vector)
    C = np.average(marray**2 + Carray,weights=rho_vector) - m**2
    return m,C


def _combine_predictions_batch(rho_vector,m_list,C_list):
    marray = np.hstack(m_list)
    m = np.average(marray,weights=rho_vector,axis=1).reshape(-1,1)
    C = np.zeros_like(C_list[0])
    for i,_ in enumerate(C_list):
        C += (C_list[i] + np.outer(m_list[i],m_list[i]))*\
              rho_vector[i]
    C = C - np.outer(m,m)
    return m,C

    
def _determine_weights(weights_term,gplist):
    W,U = weights_term
    loglikelihoods = np.array([gp.loglikelihood.item() for gp in gplist])
    loglikelihoods = loglikelihoods - np.max(loglikelihoods) #To avoid overflow
    likelihoods = np.exp(loglikelihoods)
    #Calculate rho
    rho = utilsla.invumatmul(U,likelihoods,trans="T")
    rho = utilsla.invumatmul(U,likelihoods,trans="N")
    rho = np.matmul(W,rho)
    rho = utilsla.invumatmul(U,likelihoods,trans="T")
    rho = utilsla.invumatmul(U,likelihoods,trans="T")
    rho = utilsla.invumatmul(U,likelihoods,trans="N")
    #Zero out negative terms
    rho[rho<=0] = 0
    if sum(rho) == 0:
        raise ValueError("All weight terms where negative")
#    if sum(rho) <= 0:
#        raise ValueError("Weights term summed up to nonpositive")
    rho = rho/np.sum(rho)
    return rho


def _get_prior_means(phisamples,positives):
    assert len(positives) == len(phisamples)
    prior_means = np.zeros(len(phisamples))
    for i,phisample in enumerate(phisamples):
        if not positives[i]:
            prior_means[i] = np.mean(phisample)
        else:
            prior_means[i] = spstats.mstats.gmean(phisample)
    prior_means[positives] = np.log(prior_means[positives])
    return prior_means