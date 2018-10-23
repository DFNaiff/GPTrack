# -*- coding: utf-8 -*-
import numpy as np

from . import gpmarginsquare
from . import utils

#
# Mixed implementation of
# "Real-Time Information Processing of
# Environmental Sensor Network Data using
# Bayesian Gaussian Processes", M. A. Osborne, S. J. Roberts
# and "Sampling for Inference in Probabilistic Models with
# Fast Bayesian Quadrature" by Michael A. Osborne
#

#==============================================================================
# MAIN CLASS
#==============================================================================
class GPTrackSquare(object):
    def __init__(self,kernel,phisamples,positives,data,
                      prior_variances = 1.0,gp_length_scales = 1.0,
                      noisekernel = "DEFAULT",
                      cov_threshold = 1e-3,
                      min_data = 20,max_data = 50,
                      verbose = 1):
        """
            kernel : kernel of the GPs (as provided by kernels.py)
                     kernel has to be a tensor product of 
                     an SphericalCorr kernel and another kernel
            noisekernel : kernel for the noise. If "DEFAULT", 
                          IIDKernel is chosen
            phisamples : list of samples for each hyperparameters,
                         [phi_1,...,phi_n],
                         with phi_i = [phi_i^(1),...,phi_i^(n)]
                         where n = kernel.nhyper + noisekernel.nhyper
            positives : [bool]*n list of positives hyperparameters
            data : data supplied to GP. If supplied, has to be of the form
                   [xdata,zdata], where xdata,zdata are two lists of data, 
                   and each data in xdata if a 2-elements list with 
                   the first element is a (integer) number corresponding 
                   to the (ith+1)-output
            prior_variances : a (positive) scalar or array
            gp_length_scales : a (positive) scalar or "ESTIMATE"
            verbose : degree of verbosity
            max_data : maximum number of datas for GPs, for downdating
        """
        self.max_data = max_data
        self.verbose = verbose
        self.xdata = data[0]
        self.zdata = data[1]
        self.nout = kernel.k1.nout #Number of outputs
        self.ndata,self.ndataout,self.datalocation = \
            _count_data(self.nout,self.xdata)
        self.gpmargin = gpmarginsquare.GPMarginSquare(kernel,phisamples,positives,
                                        prior_variances=prior_variances,
                                        gp_length_scales=gp_length_scales,
                                        noisekernel=noisekernel,
                                        verbose=verbose,
                                        data=data)
                
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
        drop_inds = _increment_data((self.xdata,self.zdata),
                                    (x_t,z_t),self.gpmargin.gplist,
                                    self.max_data,
                                    self.ndata,self.ndataout,
                                    self.datalocation)
        for i in range(self.gpmargin.eta):
            # Upgrades the GPs, revising covariance matrix,
            # data-dependent term to allow for dropped data
            self.gpmargin.gplist[i].downdate_batch(drop_inds)
            # Upgrades the GPs, revising covariance matrix,
            # data-dependent term and likelihoods to allow for added data
            self.gpmargin.gplist[i].update_batch([x_t,z_t])
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
        return self.gpmargin.predict(xs)

#==============================================================================
# AUXILIARY FUNCTIONS
#==============================================================================
def _count_data(nout,xdata):
    ndata = 0
    ndataout = [0]*nout
    datalocation = [0]*len(xdata)
    for i,x in enumerate(xdata):
        ndata += 1
        ndataout[x[0]] += 1
        datalocation[i] = x[0]
    return ndata,ndataout,datalocation
    

def _increment_data(current_data,new_data,gplist,
                   max_data,
                   ndata,ndataout,datalocation):
    """
        current_data : (xdata,zdata) tuple of new data
        new_data : (x_t,z_t) tuple of new data
        gplist : list of gps
        max_data : maximum of data
        ndata : number of data
        ndataout : list of number of data for each output
        datalocation : list of location of data
        returns: xdata : new xdata
                 zdata : new zdata
                 drop_inds : list of indexes to be dropped from GPs
    """
    xdata,zdata = current_data
    x_t,z_t = new_data
    xdata = xdata + x_t
    zdata = zdata + z_t
    ndatanew,ndataoutnew,datalocationnew = _count_data(len(ndataout),x_t)
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

