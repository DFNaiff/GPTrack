#import sys
#sys.path.append("../..")
#
#import numpy as np
#import torch
#
#from src import kernels,utilstorch
#
#ls = [1.0,2.0,3.0]
#thetas = [np.pi/2,np.pi/3,np.pi/4]
#
#params = [torch.tensor(p) for p in ls + thetas]
#print(params)
#kernel = kernels.SphericalCorr(3)
#kernel.initialize(params)

import numpy as np

def sample_params(hparams,bounds):
    lb,ub = list(zip(*bounds))
    lb,ub = np.array(lb),np.array(ub)
    n = len(hparams)
    return list(np.random.random(n)*(ub - lb) + lb)