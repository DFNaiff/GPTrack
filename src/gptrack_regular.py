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
# Mixed implementation of
# "Real-Time Information Processing of
# Environmental Sensor Network Data using
# Bayesian Gaussian Processes", M. A. Osborne, S. J. Roberts
# and "Sampling for Inference in Probabilistic Models with
# Fast Bayesian Quadrature" by Michael A. Osborne
#

def adjust_mean_hyperparam(phisamples,ind,alpha,gplist,p):
    #mean has no transformation
    for j,phi in enumerate(phisamples[ind]):
        indexes = utils.product_list_access(p,ind,j)
        g = 0.0
        for i in indexes:
            g += gplist[i]._dmean_loglikelihood()
        g = g/len(gplist)
        phisamples[ind][j] = phi + alpha*g


def adjust_noisevar_hyperparam(phisamples,ind,alpha,gplist,p):
    #noise_var is positive
    for j,phi in enumerate(phisamples[ind]):
        indexes = utils.product_list_access(p,ind,j)
        g = 0.0
        for i in indexes:
            g += gplist[i]._dnoisevar_loglikelihood()
        g = phi*g/len(gplist)
        phisamples[ind][j] = phi*np.exp(alpha*g)


def adjust_kernel_hyperparam(phisamples,ind,alpha,gplist,p,positive):
    if not positive:
        for j,phi in enumerate(phisamples[ind]):
            indexes = utils.product_list_access(p,ind,j)
            g = 0.0
            for i in indexes:
                g += gplist[i]._dkernel_loglikelihood(ind)
            g = g/len(gplist)
            phisamples[ind][j] = phi + alpha*g
    if positive:
        for j,phi in enumerate(phisamples[ind]):
            indexes = utils.product_list_access(p,ind,j)
            g = 0.0
            for i in indexes:
                g += gplist[i]._dkernel_loglikelihood(ind)
            g = phi*g/len(gplist)
            phisamples[ind][j] = phi*np.exp(alpha*g)


def adjust_hyperparams(phisamples,gplist,adjustables,positives,alpha):
    p = [len(phisample) for phisample in phisamples]
    for ind,adj in enumerate(adjustables):
        if adj:
            if ind == len(phisamples) - 2: #adjust mean
                adjust_mean_hyperparam(phisample,ind,alpha,gplist,p)
            elif ind == len(phisamples) - 1: #adjust noise_var
                adjust_noisevar_hyperparam(phisample,ind,alpha,gplist,p)
            else: #adjust some kernel paramater
                positive = positives[ind]
                adjust_kernel_hyperparam(phisample,ind,alpha,
                                         gplist,p,positive)
    return phisamples


def calc_gp_prior_variances(gplist,philist):
    raise NotImplementedError


def calculate_weights_term(phisamples,prior_variances,gp_prior_variances,
                           positives,prior_means=0.0):
    """
        Calculates the weight term matrix (actually, returns 
        the necessary factors)
        phisamples : list of samples for each hyperparameter dimension, 
                     [[phi_1^(1),...,phi_1^(m_1)],...,
                      [phi_n^(1),...,phi_n^(m_n)]]
        prior_variances : a (positive) scalar or n-sized array
        gp_prior_variances : a (positive) scalar or n-sized array
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
    if type(gp_prior_variances) != np.ndarray: # A scalar was supplied
        gp_prior_variances = gp_prior_variances*np.ones(n)
    #Make multivariate gaussians
    mvns = []
    for i in range(n):
        lambd = prior_variances[i]
        nu = prior_means[i]
        w = gp_prior_variances[i]
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
                                l = np.sqrt(gp_prior_variances[0]))
    A = utils.binary_function_matrix(calni,trphisamples[0])
    K = utils.binary_function_matrix(kerneli,trphisamples[0])
    B = utilsla.spla.cholesky(K,lower=False)
    W = A.copy()
    U = B.copy()
    # Other matrices
    for i,_ in enumerate(trphisamples[1:],1):
        calni = functools.partial(caln,i=i)
        kerneli = functools.partial(utils.sqexp,
                                    l = np.sqrt(gp_prior_variances[i]))
        A = utils.binary_function_matrix(calni,trphisamples[i])
        K = utils.binary_function_matrix(kerneli,trphisamples[i])
        B = utilsla.spla.cholesky(K,lower=False)
        W = np.kron(W,A)
        U = np.kron(U,B)
    return W,U


def combine_predictions_single(rho_vector,m_list,C_list):
    marray = np.array(m_list)
    Carray = np.array(C_list)
    m = np.average(marray,weights=rho_vector)
    C = np.average(marray**2 + Carray,weights=rho_vector) - m**2
    return m,C


def combine_predictions_batch(rho_vector,m_list,C_list):
    marray = np.array(m_list)
    m = np.average(marray,weights=rho_vector,axis=0)
    C = np.zeros_like(C_list[0])
    for i,_ in enumerate(C_list):
        C += (C_list[i] + np.outer(m_list[i],m_list[i]))*\
              rho_vector[i]
    C = C - np.outer(m,m)
    return m,C


def count_data(nout,xdata):
    ndata = 0
    ndataout = [0]*nout
    datalocation = [0]*len(xdata)
    for i,x in enumerate(xdata):
        ndata += 1
        ndataout[x[0]] += 1
        datalocation[i] = x[0]
    return ndata,ndataout,datalocation
    
    
def determine_weights(weights_term,gplist):
    W,U = weights_term
    loglikelihoods = np.array([gp.loglikelihood for gp in gplist])
    loglikelihoods = loglikelihoods - np.max(loglikelihoods) #To avoid overflow
    likelihoods = np.exp(loglikelihoods)
    #Calculate rho
    rho = utilsla.invumatmul(U,likelihoods,trans="T")
    rho = utilsla.invumatmul(U,likelihoods,trans="N")
    rho = np.matmul(W,rho)
    rho = utilsla.invumatmul(U,likelihoods,trans="T")
    rho = utilsla.invumatmul(U,likelihoods,trans="T")
    rho = utilsla.invumatmul(U,likelihoods,trans="N")
    #Normalize rho
    print(len(rho[rho<0])/len(rho))
    rho[rho<=0] = 0 # FIXME : this shouldn't be necessary at all. Deeply wrong.
    rho = rho/np.sum(rho)
    return rho


def increment_data(current_data,new_data,gplist,
                   min_data,max_data,cov_threshold,
                   ndata,ndataout,datalocation):
    """
        current_data : (xdata,zdata) tuple of new data
        new_data : (x_t,z_t) tuple of new data
        gplist : list of gps
        min_data : minimum of data
        max_data : maximum of data
        cov_threshold : covariance threshold
        ndata : number of data
        ndataout : list of number of data for each output
        datalocation : list of location of data
        returns: xdata : new xdata
                 zdata : new zdata
                 drop_inds : list of indexes to be dropped from GPs
    """
    # TODO : Use cov_threshold and min_data
    xdata,zdata = current_data
    x_t,z_t = new_data
    xdata = xdata + x_t
    zdata = zdata + z_t
    ndatanew,ndataoutnew,datalocationnew = count_data(len(ndataout),x_t)
    ndatanew = ndata + ndatanew
    ndataoutnew = [sum(x) for x in zip(ndataout,ndataoutnew)]
    datalocationnew = datalocation + datalocationnew
    drop_inds = []
    ndrop_outs = [0]*len(ndataout)
    while ndatanew > max_data: # Choose the data to be removed
        drop_out = np.argmax(ndataoutnew)
        ndataoutnew[drop_out] -= 1
        ndrop_outs[drop_out] += 1
        drop_ind = utils.nth_index(datalocationnew,
                                   drop_out,ndrop_outs[drop_out])
        drop_inds.append(drop_ind)
        ndatanew -= 1
    drop_inds.sort()
    for drop_ind in drop_inds[::-1]: # Removing data
        xdata.pop(drop_ind)
        zdata.pop(drop_ind)
        datalocationnew.pop(drop_ind)
    return xdata,zdata,ndatanew,ndataoutnew,datalocationnew,drop_inds


def get_prior_means(phisamples,positives):
    assert len(positives) == len(phisamples)
    prior_means = np.zeros(len(phisamples))
    for i,phisample in enumerate(phisamples):
        if not positives[i]:
            prior_means[i] = np.mean(phisample)
        else:
            prior_means[i] = spstats.mstats.gmean(phisample)
    prior_means[positives] = np.log(prior_means[positives])
    return prior_means


#==============================================================================
# MAIN CLASS
#==============================================================================
class GPTrackMO(object):
    def __init__(self,kernel,phisamples,positives,
                      adjustables = False,weights_term_construction = "new",
                      prior_variances = 1.0,gp_prior_variances = 1.0,
                      noisekernel = "DEFAULT",
                      cov_threshold = 1e-3,
                      min_data = 20,max_data = 50,
                      verbose = 1,data=None):
        """
            kernel : kernel of the GPs (as provided by kernels.py)
                     kernel has to be a tensor product of 
                     an SphericalCorr kernel and another kernel
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
            gp_prior_variances : a (positive) scalar or "ESTIMATE"
            cov_threshold : minimal covariance
            min_data : minimum number of datas for GPs, for downdating
            max_data : maximum number of datas for GPs, for downdating
            verbose : degree of verbosity
            data : data supplied to GP. If supplied, has to be of the form
                   [xdata,zdata], where xdata,zdata are two lists of data, 
                   and each data in xdata if a 2-elements list with 
                   the first element is a (integer) number corresponding 
                   to the (ith+1)-output
        """
        self.kernel = kernel
        if noisekernel == "DEFAULT":
            self.noisekernel = kernels.IIDNoiseKernel()
        else:
            self.noisekernel = noisekernel
        self.phisamples = phisamples
        self.positives = positives
        if adjustables == False:
            self.adjustables = False*len(self.positives)
        else:
            self.adjustables = adjustables
        self.weights_term_construction = weights_term_construction
        self.prior_variances = prior_variances
        self.gp_prior_variances = gp_prior_variances
        self.cov_threshold = cov_threshold
        self.min_data = min_data
        self.max_data = max_data
        self.verbose = verbose
        self._initialize(data)
    
    def add_data(self,x_t,z_t):
        """
            x_t = [x_t1,x_t2,...,x_tn] list of inputs
            z_t = [x_t1,x_t2,...,x_tn] list of outputs
        """
        # Add new datum (index add_ind), and drop old data 
        # (indices drop_inds) as necessary
        # TODO : Data dropping based on covariance
        self.xdata,self.zdata,\
        self.ndata,self.ndataout,self.datalocation,\
        drop_inds = increment_data((self.xdata,self.zdata),
                                   (x_t,z_t),self.gplist,
                                   self.min_data,self.max_data,
                                   self.cov_threshold,
                                   self.ndata,self.ndataout,
                                   self.datalocation)
        for i in range(self.eta):
            # Upgrades the GPs, revising covariance matrix,
            # data-dependent term to allow for dropped data
            self.gplist[i].downdate_batch(drop_inds)
            # Upgrades the GPs, revising covariance matrix,
            # data-dependent term and likelihoods to allow for added data
            self.gplist[i].update_batch([x_t,z_t])
        self.rho_vector = determine_weights(self.weights_term,self.gplist)
        if self.verbose >= 1:
            print("ndata: %i",self.ndata)
        return
    
    def predict(self,xs):
        """
            xs = [x_1,...,x_m] list of desired predictions
            returns : 
                means = [mean_1,...,mean_m] list of predictions means
                covs = [cov_1,...,cov_m] list of predicitions vars
                
        """
        if type(xs) not in [list,tuple]:
            return self._predict_single(xs)
        else:
            return self._predict_batch(xs)
    
    def update_hyperparameters(self,alpha=0.01):
        self.phisamples = adjust_hyperparams(self.phisamples,
                                             self.gplist,
                                             self.adjustables,
                                             self.positives,
                                             alpha)
        self._initialize([self.xdata,self.zdata])
        return

    
    def _initialize(self,data=None):
        #Total number of samples
        self.eta = np.product([len(phisample) 
                               for phisample in self.phisamples])
        self.nout = self.kernel.k1.nout #Number of outputs
        self.prior_means = get_prior_means(self.phisamples,
                                           self.positives)
        if not data: #If data is not supplied
            self.xdata = []
            self.zdata = []
            self.ndata = 0
            self.ndataout = [0]*self.nout
        else: #If data is supplied
            self.xdata = copy.copy(data[0])
            self.zdata = copy.copy(data[1])
            self.ndata,self.ndataout,self.datalocation = \
                count_data(self.nout,self.xdata)
        if self.verbose >= 1:
            print("Setting GPS")
        self.gplist = [None]*self.eta
        phiiter = itertools.product(*self.phisamples)
        if self.gp_prior_variances == "ESTIMATE":
            philist = list(phiiter)
        for i,phisample in enumerate(phiiter):
            phisample = np.array(phisample)
            # Supply ith gaussian process with the mean function meanfunc 
            # and covariance function cov func
            self.gplist[i] = gpobject.GPObject(self.kernel,self.noisekernel,
                                               phisample)
            if data:
                self.gplist[i].change_data(data)
        # Estimate the GP prior variances if so desired
        if self.gp_prior_variances == "ESTIMATE":
            if not data:
                raise AssertionError
            else:
                self.gp_prior_variances = calc_gp_prior_variances(self.gplist,
                                                                  philist)
        # Calculate the weights
        if self.verbose >= 1:
            print("Calculating weight terms")
        if self.weights_term_construction == "new":
            self.weights_term = calculate_weights_term(self.phisamples,
                                                       self.prior_variances,
                                                       self.gp_prior_variances,
                                                       self.positives,
                                                       self.prior_means)
        else:
            self.weights_term = calculate_weights_term_old(self.phisamples,
                                                           self.prior_variances,
                                                           self.gp_prior_variances,
                                                           self.positives,
                                                           self.prior_means)
        self.rho_vector = determine_weights(self.weights_term,self.gplist)
        if self.verbose >= 1 : 
            print("Initialized")
    
    def _predict_single(self,x):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict(x)
            m_list[i] = m_i
            C_list[i] = C_i
        m,C = combine_predictions_single(self.rho_vector,m_list,C_list)
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
        m,C = combine_predictions_batch(self.rho_vector,m_list,C_list)
        if not retvar:
            return m,C
        else:
            return m,np.diag(C)