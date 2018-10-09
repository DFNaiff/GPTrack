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
# GP with marginalization according to a mix of
# "Real-Time Information Processing of
# Environmental Sensor Network Data using
# Bayesian Gaussian Processes", M. A. Osborne, S. J. Roberts
# and "Sampling for Inference in Probabilistic Models with
# Fast Bayesian Quadrature" by Michael A. Osborne
#

class GPMarginSquare(object):
    def __init__(self,kernel,phisamples,positives,
                      prior_variances = 1.0,gp_length_scales = 1.0,
                      noisekernel = "DEFAULT",verbose=1,
                      data=None):
        """
            kernel : kernel of the GPs
            noisekernel : kernel for the noise. If "DEFAULT", 
                          IIDKernel is chosen
            phisamples : list of samples for each hyperparameters,
                         [phi_1,...,phi_n],
                         with phi_i = [phi_i^(1),...,phi_i^(n)]
                         where n = kernel.nhyper + noisekernel.nhyper
            positives : [bool]*n list of positives hyperparameters
            adjustable : [bool]*n list of hyperparameters to be adjusted
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
    
    
    def initialize(self,data=None):
        #Total number of samples
        self.eta = len(self.phisamples)
        #TODO : VERY NON BAYESIAN
        self.prior_means = _get_prior_means(self.phisamples,self.positives)
        self.xdata = copy.copy(data[0])
        self.zdata = copy.copy(data[1])
        if self.verbose >= 1:
            print("Setting GPS")
        self._set_gps([self.xdata,self.zdata])
        # Calculate the weights
        if self.verbose >= 1:
            print("Calculating weight terms")
        self.weights_term = _calculate_weights_term(self.phisamples,
                                                   self.positives,
                                                   self.gp_length_scales,
                                                   self.prior_means,
                                                   self.prior_variances,
                                                   jitter = 1e-10)

        if self.verbose >= 1 : 
            print("Initialized")
        self._initialized = True
    
    def _predict_single(self,x):
        raise NotImplementedError
        return m,C
    
    def _predict_batch(self,xs,retvar = True):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        log_likelihoods = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict_batch(xs,retdiag=False)
            m_list[i] = m_i
            C_list[i] = C_i
            log_likelihoods[i] = self.gplist[i].loglikelihood
        m,C = _combine_predictions_batch(self.weights_term,
                                                log_likelihoods,
                                                m_list,C_list)
        if not retvar:
            return m,C
        else:
            return m,np.diag(C)
    
    def _check_initialized(self):
        if not self._initialized:
            raise TypeError #TODO : Change error
    
    #Initialization functons
    def _set_gps(self,data):
        self.gplist = [None]*self.eta
        for i,phisample in enumerate(self.phisamples):
            phisample = np.array(phisample)
            # Supply ith gaussian process with the mean function meanfunc 
            # and covariance function cov func
            self.gplist[i] = gpobject.GPObject(self.kernel,self.noisekernel,
                                               phisample,data)
#==============================================================================
# AUXILIARY FUNCTIONS
#==============================================================================
def _calculate_weights_term(phisamples,positives,
                            gp_length_scales=1.0,
                            prior_means=0.0,prior_variances=20.0,
                            jitter = 1e-10):
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
        returns a m_1*...*m_n sized square matrix
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
    # Multivariate normal used for W
    # TODO: change for more efficient manner
    l2 = gp_length_scales**2
    C = utilsla.block([[np.diag(prior_variances + l2),np.diag(prior_variances)],
                       [np.diag(prior_variances),np.diag(prior_variances + l2)]])
    mvn = spstats.multivariate_normal(mean = np.hstack([prior_means]*2),
                                      cov=C)
    wfunc = lambda phi_i,phi_j : mvn.pdf(np.hstack([phi_i,phi_j]))
    kfunc = functools.partial(utils.sqexp,l=gp_length_scales)
    K = utils.binary_function_matrix(kfunc,phisamplestr)
    K = K + np.identity(K.shape[0])*jitter
    U = utilsla.spla.cholesky(K,lower=False)
    invK = utilsla.inverse_cholesky_upper(U)
    W = utils.binary_function_matrix(wfunc,phisamplestr)
    M = np.matmul(invK,np.matmul(W,invK))
    return M


def _make_wij(phi_i,phi_j,l2,lambd2,nu): #TODO : Check again
    #Deprecated
    V2 = 1.0/(2/l2 + 1/lambd2)
    C = (phi_i + phi_j)/l2 + nu**2/lambd2 - \
         1.0/(2/l2 + 1/lambd2)*((phi_i + phi_j)/l2 + nu/lambd2)
    mij = np.prod(np.sqrt(V2/lambd2)*np.exp(-0.5*C))
    return mij


def _combine_predictions_batch(weights_term,
                                     loglikelihoods,
                                     m_list,C_list):
    n = len(loglikelihoods)
    d = len(m_list[0])
#    print(loglikelihoods)
    loglikelihoods = loglikelihoods - np.max(loglikelihoods) #To avoid overflow
    likelihoods = np.exp(loglikelihoods).reshape(-1,1)
    m_array = np.transpose(np.hstack(m_list))
    C_array = np.transpose(np.dstack(C_list),(2,0,1))
    lD_array = np.linalg.slogdet(C_array)[1].reshape(-1,1)
    I_array = np.linalg.inv(C_array)
    #Tile and repeat trick
#    print(C_array)
#    print(lD_array)
    M1 = np.repeat(m_array,n,axis=0)
    M2 = np.tile(m_array,[n,1])
    C1 = np.repeat(C_array,n,axis=0)
    C2 = np.tile(C_array,[n,1,1])
    L1 = np.repeat(likelihoods,n,axis=0)
    L2 = np.tile(likelihoods,[n,1])
    lD1 = np.repeat(lD_array,n,axis=0)
    lD2 = np.tile(lD_array,[n,1])
    I1 = np.repeat(I_array,n,axis=0)
    I2 = np.tile(I_array,[n,1,1])
    #Make weights
    #TODO : Make it by cholesky factorization
    #TODO : Bad naming
    invsumcov = np.linalg.inv(C1 + C2)
    diffM = (M1 - M2).reshape(n**2,d,1)
    expoent = -0.25*np.matmul(diffM.transpose(0,2,1),
                              np.matmul(invsumcov,diffM)).reshape(-1,1)
    ldetsumcov = np.linalg.slogdet(C1 + C2)[1].reshape(-1,1)
    Cexp = np.exp(expoent)
    weights_vectorized = weights_term.reshape(-1,1)
    detterm = np.exp(0.25*(lD1 + lD2) - 0.5*(ldetsumcov))
    w = weights_vectorized*detterm*np.sqrt(L1*L2)*Cexp
    w = w/np.sum(w)
    #Means
    M1b,M2b = M1.reshape(n**2,d,1),M2.reshape(n**2,d,1)
    invsumcov2 = np.linalg.inv(I1 + I2)
    M = np.matmul(invsumcov2,np.matmul(I1,M1b) + \
                  np.matmul(I2,M2b)).reshape(n**2,d)
#    print(M)
    mean = np.sum(w*M,axis=0).reshape(-1,1)
    #Covariances
    covterm = 2*invsumcov2*np.matmul(M.reshape(n**2,d,1),
                                     M.reshape(n**2,1,d))
    wbroadcasted = np.tile(w[np.newaxis].transpose(1,0,2),[1,d,d])
    cov = np.sum(wbroadcasted*covterm,axis=0) - np.outer(mean,mean)
    return mean,cov

def _get_prior_means(phisamples,positives):
    phigrid = list(np.array(phisamples).transpose())
    assert len(positives) == len(phigrid)
    prior_means = np.zeros(len(phigrid))
    for i,phisample in enumerate(phigrid):
        if not positives[i]:
            prior_means[i] = np.mean(phisample)
        else:
            prior_means[i] = spstats.mstats.gmean(phisample)
    prior_means[positives] = np.log(prior_means[positives])
    return prior_means

#==============================================================================
# DEPECRATED FOR NOW 
#==============================================================================
def _combine_predictions_batch2(weights_term,
                                     loglikelihoods,
                                     m_list,C_list):
    #TODO : Eficiency
    m_list = [m.flatten() for m in m_list]     
    loglikelihoods = loglikelihoods - np.max(loglikelihoods) #To avoid overflow
    likelihoods = np.exp(loglikelihoods)
    N = len(likelihoods)
    iterator = itertools.product(range(N),range(N))
    sum_weights = 0.0
    mean = 0.0
    cov = 0.0
    for i,j in iterator:
        weight = _make_weight(m_list[i],m_list[j],
                             C_list[i],C_list[j],
                             likelihoods[i],likelihoods[j],
                             weights_term[i][j])
        invcov_i = utilsla.spla.inv(C_list[i])
        invcov_j = utilsla.spla.inv(C_list[j])
        covij = utilsla.spla.inv(invcov_i + invcov_j)
        meanij = np.matmul(covij,(np.matmul(invcov_i,m_list[i]) + \
                                  np.matmul(invcov_j,m_list[j])))
        print(meanij)
        mean += weight*meanij
        cov += weight*(covij + np.outer(meanij,meanij))
        sum_weights += weight
    mean = mean/sum_weights
    cov = cov/sum_weights - np.outer(mean,mean)
    return mean,cov


def _make_weight(mi,mj,covi,covj,li,lj,Mij):
    C = (utilsla.spla.det(covi)*utilsla.spla.det(covj))**(0.25)
    D = 1.0/np.sqrt(utilsla.spla.det(covi + covj))*\
        np.exp(-0.25*utilsla.bilinear_form(mi - mj,
                                          utilsla.spla.inv(covi + covj),
                                          mi - mj))
#    print(C*D,np.sqrt(li*lj),Mij)
    result = C*D*np.sqrt(li*lj)*Mij
    return result

def _calculate_weights_term_old(phisamples,positives,
                            gp_length_scales=1.0,
                            prior_means=0.0,prior_variances=20.0,
                            jitter = 1e-6):
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
        returns a m_1*...*m_n sized square matrix
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
    # Multivariate normal used in caln
    # TODO: change for more efficient manner
    l2 = gp_length_scales**2
    kfunc = functools.partial(utils.sqexp,l=gp_length_scales)
    wfunc = functools.partial(_make_wij,l2=l2,
                              lambd2 = prior_variances,
                              nu = prior_means)
    K = utils.binary_function_matrix(kfunc,phisamplestr)
    K = K + np.identity(K.shape[0])*jitter
    U = utilsla.spla.cholesky(K,lower=False)
    invK = utilsla.inverse_cholesky_upper(U)
    W = utils.binary_function_matrix(wfunc,phisamplestr)
    M = np.matmul(invK,np.matmul(W,invK))
    return M


def _make_wij(phi_i,phi_j,l2,lambd2,nu): #TODO : Check again
    V2 = 1.0/(2/l2 + 1/lambd2)
    C = (phi_i + phi_j)/l2 + nu**2/lambd2 - \
         1.0/(2/l2 + 1/lambd2)*((phi_i + phi_j)/l2 + nu/lambd2)
    mij = np.prod(np.sqrt(V2/lambd2)*np.exp(-0.5*C))
    return mij