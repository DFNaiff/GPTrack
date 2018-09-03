# -*- coding: utf-8 -*-

#TODO : Don't repeat yourself should be obeyed.
#       But I don't see how (choose_next_single and choose_next_mo)

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from . import gpobject
from . import kernels
from . import utils
from . import utilsla


def choose_next_gptrack(gptrack,t0,t1,
                        max_error,confidence=0.95):
    """
        Choose next measurement point for GPTrack
    """
    loglikelihoods = np.array([gp.loglikelihood for gp in gptrack.gplist])
    ind = np.argmax(loglikelihoods)
    gpi = gptrack.gplist[ind] #GP with largest likelihood, used for estimation
    return choose_next_mo(gpi,t0,t1,max_error,confidence)


def choose_next_mo(gp,t0,t1,
                   max_error,confidence=0.95):
    """
        GP should be multiple output
        gp : GPObject
        t0 : current measured time
        t1 : some next step
        max_error : maximum tolerable error (within confidence range)
                    may be a number or a gp.nout-sized array
        confidence : confidence range
        
        returns:
            tnews = [tnew1,...,tnewn] list of when to measure each output
    """
#    Since GPs (and other methods) are bad extrapolators,
#    the technique is the following:
#    sample points t0,t1,t2,t3,...
#    where t_i = 2*(t_{i-1} - t_{i-2}) + t_{i-1} 
#    until t_i extrapolates covariance. 
#    Then find zero between t_{i-1} and t_i using GP
#    Do this for each of the output time series, 
#    then choose the earliest_t0
    nout = gp.kernel.k1.nout
    tnews = [None]*nout #List to put tnew values
    try: #Checking whether max_error is valid
        assert len(max_error) == nout
    except TypeError:
        max_error = max_error*np.ones(nout)
    #Calculating tolerable error
    covlims = max_cov(max_error,confidence)
    #Checking whether there is too much uncertainty already
    covs0 = np.diag(gp.predict_batch([[i,t0] for i in range(nout)])[1])
    for i,cov0 in enumerate(covs0):
        if cov0 >= covlims[i]: #Measure right away
            tnews[i] = t0
    #Choosing points
    for i in range(nout):
        if tnews[i] != None: #Already measures right away
            continue
        def f(t):
            return gpchoice.predict(t)[0] - covlim
        covlim = covlims[i]
        tdata = [t0,t1]
        is_above = False
        while not is_above:
            covi = gp.predict([i,tdata[-1]])[1]
            if covi > covlim:
                is_above = True
            else:
                tdata.append(2*(tdata[-1] - tdata[-2]) + tdata[-1])
        ta,tb = tdata[-1],tdata[-2]
        #Getting more points for data
        tdata.extend(list(np.linspace(tdata[-2],tdata[-1],6)[1:-1]))
        tdatab = [[i,t] for t in tdata]
        #Preparing GP        
        zdata = np.diag(gp.predict_batch(tdatab)[1]) #Measured covs
        choicekernel = kernels.Constant()*kernels.IsoRBF(dim=1)
        choicenoisekernel = kernels.IIDNoiseKernel()
        print(i,'ok')
        gpchoice = gpobject.GPObject(choicekernel,choicenoisekernel,
                    np.array([1.0,1.0,1e-6]),[tdata,zdata])
        gpchoice = gpchoice.optimize([True,True,True],[True,True,False],
                                     [None,None,"log"],verbose=0)
        #Calculating t
        try:
            tnewi = brentq(f,ta,tb)
        except:
            tnewi = ta
            print("Error in brentq. Choosing ta")
        tnews[i] = tnewi
    return tnews


def choose_next_single(gp,t0,t1,
                       max_error,confidence=0.95):
    """
        Since GPs (and other methods) are bad extrapolators,
        the technique is the following:
        sample points t0,t1,t2,t3,...
        where t_i = 2*(t_{i-1} - t_{i-2}) + t_{i-1} 
        until t_i extrapolates covariance. 
        Then find zero between t_{i-1} and t_i using GP
    """
    def f(t):
        return gpchoice.predict(t)[0] - covlim
    #Calculating tolerable error
    covlim = max_cov(max_error,confidence)
    #Checking whether there is too much uncertainty
    cov0 = gp.predict(t0)[1]
    if cov0 >= covlim:
        return t0
    #Choosing points
    tdata = [t0,t1]
    is_above = False
    while not is_above:
        covi = gp.predict(tdata[-1])[1]
        if covi > covlim:
            is_above = True
        else:
            tdata.append(2*(tdata[-1] - tdata[-2]) + tdata[-1])
    ta,tb = tdata[-1],tdata[-2]
    #Getting more points for data
    tdata.extend(list(np.linspace(tdata[-2],tdata[-1],6)[1:-1]))
    zdata = np.diag(gp.predict_batch(tdata)[1]) #Measured covs
    choicekernel = kernels.Constant()*kernels.IsoRBF(dim=1)
    choicenoisekernel = kernels.IIDNoiseKernel()
    gpchoice = gpobject.GPObject(choicekernel,choicenoisekernel,
                np.array([1.0,1.0,1e-6]),[tdata,zdata])
    gpchoice = gpchoice.optimize([True,True,True],[True,True,True],
                                 [None,None,"log"],verbose=0)
    tnew = brentq(f,ta,tb)
    return tnew


def max_cov(max_error,confidence):
    return (max_error/(norm.ppf((confidence + 1)/2.0)))**2