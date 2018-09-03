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


def calculate_weights_term(phisamples,positives,
                            gp_length_scales=1.0,
                               prior_means=0.0,prior_variances=20.0,
                               jitter = 1e-6):
    """
        Calculates the weight term matrix without grid 
        phisamples : [phi_1,...,phi_eta] list of hyperparameters samples,
                     where phi_i = [phi_i^1,...,phi_i^n]
        prior_variances : a (positive) scalar (LATER: also arrays)
        gp_prior_variances : a (positive) scalar (LATER: also arrays)
        positives : [bool]*n list of parameters that are positive
        prior_means : a scalar or n-sized array
        returns a m_1*...*m_n sized square matrix
    """
    phisamples = list(map(np.array,itertools.product(*phisamples)))
    eta = len(phisamples)
    n = len(phisamples[0])
    phisamplestr = [None]*eta
    for i,phisample in enumerate(phisamples):
        # Apply log to positive hyperparameters
        phisampletr = phisample.copy()
#        print(phisample[positives])
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
    wvec = gp_length_scales**2
    C = utilsla.block([[np.diag(prior_variances + wvec),np.diag(wvec)],
                       [np.diag(wvec),np.diag(prior_variances + wvec)]])
    mvn = spstats.multivariate_normal(mean = np.hstack([prior_means]*2),
                                      cov=C)
    def k(phi_i,phi_j):
        return utils.sqexp(phi_i,phi_j,gp_length_scales)
    def caln(phi_i,phi_j):
        return mvn.pdf(np.hstack([phi_i,phi_j]))
    K = utils.binary_function_matrix(k,phisamplestr)
    K = K + np.identity(K.shape[0])*jitter
    U = utilsla.spla.cholesky(K,lower=False)
    invK = utilsla.inverse_cholesky_upper(U)
    W = utils.binary_function_matrix(caln,phisamplestr)
    M = np.matmul(invK,np.matmul(W,invK))
    return M


def calculate_weights_term_grid(phisamples,positives,
                           gp_length_scales=1.0,
                           prior_means=0.0,prior_variances=20.0):
    """
        Calculates the weight term matrix with samples in a grid
        phisamples : list of samples for each hyperparameter dimension, 
                     [[phi_1^(1),...,phi_1^(m_1)],...,
                      [phi_n^(1),...,phi_n^(m_n)]]
        positives : [bool]*n list of positives hyperparameters
        gp_length_scales : a (positive) scalar or array
                             SHOULD BE SET BY MAXIMIZING LOG LIKELIHOOD
        prior_means : a scalar or n-sized array
        prior_variances : a (positive) scalar or n-sized array
                          SHOULD IDEALLY BE SET BEFOREHAND MANUALLY,
                          IN A FLAT MANNER
        returns a m_1*...*m_n sized square matrix
    """
    n = len(phisamples)
    trphisamples = copy.deepcopy(phisamples)
    for i,_ in enumerate(phisamples): #Transform positives variables
        if positives[i]:
            trphisamples[i] = np.log(phisamples[i])
    if type(gp_length_scales) != np.ndarray: # A scalar was supplied
        gp_length_scales = gp_length_scales*np.ones(n)
    if type(prior_variances) != np.ndarray: # A scalar was supplied
        prior_variances = prior_variances*np.ones(n)
    if type(prior_means) != np.ndarray: # A scalar was supplied
        prior_means = prior_means*np.ones(n)
    # Makes multivariates normal distributions used in caln
    mvns = []
    for i in range(n):
        lambd = prior_variances[i]
        nu = prior_means[i]
        w = gp_length_scales[i]
        m = np.array([nu,nu])
        C = np.array([[lambd + w**2, lambd],
                      [lambd, lambd + w**2]])
        mvn = spstats.multivariate_normal(mean=m,cov=C)
        mvns.append(mvn)
    def caln(phi_i,phi_j,i):
        return mvns[i].pdf(np.hstack([phi_i,phi_j]))
    def makeA(phisamplesi,i):
        ki = functools.partial(utils.sqexp,l = gp_length_scales[i])
        K = utils.binary_function_matrix(ki,phisamplesi)
        invK = utilsla.inverse_cholesky(K)
        calni = functools.partial(caln,i=i)
        N = utils.binary_function_matrix(calni,phisamplesi)
        A = np.matmul(invK,np.matmul(N,invK))
        return A
    # Calculate Kronecker product
    # First matrix
    A = makeA(trphisamples[0],0)
    M = A.copy()
    # Other matrices
    for i,_ in enumerate(trphisamples[1:],1):
        A = makeA(trphisamples[i],i)
        M = np.kron(M,A)
    return M


def combine_predictions_single_square(weights_term,
                                      loglikelihoods,
                                      m_list,C_list):
    #TODO : Unnest functions
    #TODO : Eficiency
    def make_weight(mi,mj,covi,covj,li,lj,Mij):
        C = (covi*covj)**(0.25)
        D = np.sqrt((covi + covj))*np.exp(-0.5*(mi - mj)**2/(covi + covj))
        result = C*D*np.sqrt(li*lj)*Mij
        return result
    loglikelihoods = loglikelihoods - np.max(loglikelihoods) #To avoid overflow
    likelihoods = np.exp(loglikelihoods)
    N = len(likelihoods)
    iterator = itertools.product(range(N),range(N))
    sum_weights = 0.0
    mean = 0.0
    cov = 0.0
    for i,j in iterator:
        weight = make_weight(m_list[i],m_list[j],
                             C_list[i],C_list[j],
                             likelihoods[i],likelihoods[j],
                             weights_term[i][j])
        invcov_i,invcov_j = 1.0/C_list[i],1.0/C_list[j]
        covij = 1.0/(invcov_i + invcov_j)
        meanij = covij*(invcov_i*m_list[i] + invcov_j*m_list[j])
        mean += weight*meanij
        cov += weight*(covij + meanij**2)
        sum_weights += weight
    mean = mean/sum_weights
    cov = cov/sum_weights - mean
    return mean,cov


def combine_predictions_batch_square(weights_term,
                                     loglikelihoods,
                                     m_list,C_list):
#    marray = np.array(m_list)
#    m = np.average(marray,weights=rho_vector,axis=0)
#    C = np.zeros_like(C_list[0])
#    for i,_ in enumerate(C_list):
#        C += (C_list[i] + np.outer(m_list[i],m_list[i]))*\
#              rho_vector[i]
#    C = C - np.outer(m,m)
#    return m,C
    def make_weight(mi,mj,covi,covj,li,lj,Mij):
        C = (utilsla.spla.det(covi)*utilsla.spla.det(covj))**(0.25)
        D = np.sqrt(utilsla.spla.det(covi) + utilsla.spla.det(covj))*\
            np.exp(-0.5*utilsla.bilinear_form(mi - mj,
                                              utilsla.spla.inv(covi + covj),
                                              mi - mj))
        result = C*D*np.sqrt(li*lj)*Mij
        return result
    loglikelihoods = loglikelihoods - np.max(loglikelihoods) #To avoid overflow
    likelihoods = np.exp(loglikelihoods)
    N = len(likelihoods)
    iterator = itertools.product(range(N),range(N))
    sum_weights = 0.0
    mean = 0.0
    cov = 0.0
    for i,j in iterator:
        weight = make_weight(m_list[i],m_list[j],
                             C_list[i],C_list[j],
                             likelihoods[i],likelihoods[j],
                             weights_term[i][j])
        invcov_i = utilsla.spla.inv(C_list[i])
        invcov_j = utilsla.spla.inv(C_list[j])
        covij = utilsla.spla.inv(invcov_i + invcov_j)
        meanij = np.matmul(covij,(np.matmul(invcov_i,m_list[i]) + \
                                  np.matmul(invcov_j,m_list[j])))
        mean += weight*meanij
        cov += weight*(covij + np.outer(meanij,meanij))
        sum_weights += weight
    mean = mean/sum_weights
    cov = cov/sum_weights - np.outer(mean,mean)
    return mean,cov

def count_data(nout,xdata):
    ndata = 0
    ndataout = [0]*nout
    datalocation = [0]*len(xdata)
    for i,x in enumerate(xdata):
        ndata += 1
        ndataout[x[0]] += 1
        datalocation[i] = x[0]
    return ndata,ndataout,datalocation


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
class GPTrackMOSquare(object):
    def __init__(self,kernel,phisamples,positives,
                      adjustables = False,ingrid = True,
                      gp_length_scales = 1.0,
                      prior_variances = 20.0,
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
            phisamples : 
                         if ingrid == True,
                         list of samples for each hyperparameter dimension, 
                         [[phi_1^(1),...,phi_1^(m_1)],...,
                          [phi_n^(1),...,phi_n^(m_n)]]
                         if ingrid == False:
                         list of samples for each hyperparameters,
                         [phi_1,...,phi_n],
                         with phi_i = [phi_i^(1),...,phi_i^(n)]
                         where n = kernel.nhyper + noisekernel.nhyper
            positives : [bool]*n list of positives hyperparameters
            adjustable : [bool]*n list of hyperparameters to be adjusted
            prior_variances : a (positive) scalar or array
            gp_length_scales : a (positive) scalar or array
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
        self.ingrid = ingrid
        self.prior_variances = prior_variances
        self.gp_length_scales = gp_length_scales
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
        if self.ingrid:
            self.eta = np.product([len(phisample) 
                                   for phisample in self.phisamples])
        else:
            self.eta = len(self.phisamples)
        self.nout = self.kernel.k1.nout #Number of outputs
        #TODO : VERY NON BAYESIAN
        if not self.ingrid:
            phigrid = list(np.array(self.phisamples).transpose())
            self.prior_means = get_prior_means(phigrid,self.positives)
        else:
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
        if self.ingrid:
            phiiter = itertools.product(*self.phisamples)
        else:
            phiiter = self.phisamples
        for i,phisample in enumerate(phiiter):
            print(phisample)
            phisample = np.array(phisample)
            # Supply ith gaussian process with the mean function meanfunc 
            # and covariance function cov func
            self.gplist[i] = gpobject.GPObject(self.kernel,self.noisekernel,
                                               phisample)
            if data:
                self.gplist[i].change_data(data)
        # Calculate the weights
        if self.verbose >= 1:
            print("Calculating weight terms")
        if self.ingrid:
            self.weights_term = calculate_weights_term_grid(self.phisamples,
                                                       self.positives,
                                                       self.gp_length_scales,
                                                       self.prior_means,
                                                       self.prior_variances)
        else:
            self.weights_term = calculate_weights_term(self.phisamples,
                                                       self.positives,
                                                       self.gp_length_scales,
                                                       self.prior_means,
                                                       self.prior_variances,
                                                       jitter = 1e-10)

        if self.verbose >= 1 : 
            print("Initialized")
    
    def _predict_single(self,x):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        log_likelihoods = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict(x)
            m_list[i] = m_i
            C_list[i] = C_i
            log_likelihoods[i] = self.gplist[i].loglikelihood
        m,C = combine_predictions_single_square(self.weights_term,
                                                log_likelihoods,
                                                m_list,C_list)
        return m,C
    
    def _predict_batch(self,xs,retvar = True):
        m_list = [None]*self.eta
        C_list = [None]*self.eta
        log_likelihoods = [None]*self.eta
        for i in range(self.eta):
            # Computes the posterior mean and variances for 
            # predictand y_star = y_tnew
            m_i,C_i = self.gplist[i].predict_batch(xs)
            m_list[i] = m_i
            C_list[i] = C_i
            log_likelihoods[i] = self.gplist[i].loglikelihood
        m,C = combine_predictions_batch_square(self.weights_term,
                                                log_likelihoods,
                                                m_list,C_list)
        if not retvar:
            return m,C
        else:
            return m,np.diag(C)