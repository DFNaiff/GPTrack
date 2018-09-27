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
class GPMarginGrid(object):
    def __init__(self,kernel,phisamples,positives,
                      weights_term_construction = "new",
                      prior_variances = 1.0,gp_length_scales = 1.0,
                      noisekernel = "DEFAULT",verbose=1,
                      data=None):
        """
            kernel : kernel of the GPs
            noisekernel : kernel for the noise. If "DEFAULT", 
                          IIDKernel is chosen
            phisamples : list of samples for each hyperparameter dimension, 
                         [[phi_1^(1),...,phi_1^(m_1)],...,
                          [phi_n^(1),...,phi_n^(m_n)]]
                         where n = kernel.nhyper + noisekernel.nhyper
            positives : [bool]*n list of positives hyperparameters
            adjustable : [bool]*n list of hyperparameters to be adjusted
            weights_term_construction : "new" or "old", depending on how to 
                                        construct
            prior_variances : a (positive) scalar or array
            gp_length_scales : a (positive) scalar or "ESTIMATE"
            verbose : degree of verbosity
            data : data supplied to GP. If supplied, has to be of the form
                   [xdata,zdata], where xdata,zdata are two lists of data, 
                   and each data in xdata if a 2-elements list with 
                   the first element is a (integer) number corresponding 
                   to the (ith+1)-output. If None, kernel is 
                   considered to not be initialized.
        """
        self.kernel = kernel
        if noisekernel == "DEFAULT":
            self.noisekernel = kernels.IIDNoiseKernel()
        else:
            self.noisekernel = noisekernel
        self.phisamples = phisamples
        self.positives = positives
        self.weights_term_construction = weights_term_construction
        self.prior_variances = prior_variances
        self.gp_length_scales = gp_length_scales
        self.verbose = verbose
        if data:
            self.initialize(data)
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
    
    def initialize(self,data):
        #Total number of samples
        self.eta = np.product([len(phisample) 
                               for phisample in self.phisamples])
        self.prior_means = _get_prior_means(self.phisamples,
                                            self.positives)

        self.xdata = copy.copy(data[0])
        self.zdata = copy.copy(data[1])
        if self.verbose >= 1:
            print("Setting GPS")
        self._set_gps([self.xdata,self.zdata])
        # Calculate the weights
        if self.verbose >= 1:
            print("Calculating weight terms")
        self.weights_term = _calculate_weights_term(self.phisamples,
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
    
    def _predict_batch(self,xs,retvar = True):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict_batch(xs)
            m_list[i] = m_i
            C_list[i] = C_i
        print(m_list,C_list)
        m,C = _combine_predictions_batch(self.rho_vector,m_list,C_list)
        if not retvar:
            return m,C
        else:
            return m,np.diag(C)
    
    def _check_initialized(self):
        if not self._initialized:
            raise TypeError #TODO : Change error
    
    #Initialization parameters
    def _set_gps(self,data):
        self.gplist = [None]*self.eta
        phiiter = itertools.product(*self.phisamples)
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
def _calculate_weights_term(phisamples,prior_variances,gp_length_scales,
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
    marray = np.array(m_list)
    Carray = np.array(C_list)
    m = np.average(marray,weights=rho_vector)
    C = np.average(marray**2 + Carray,weights=rho_vector) - m**2
    return m,C


def _combine_predictions_batch(rho_vector,m_list,C_list):
    marray = np.array(m_list)
    m = np.average(marray,weights=rho_vector,axis=0)
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
    if max(rho) == 0:
        raise ValueError("All weight terms where negative")
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