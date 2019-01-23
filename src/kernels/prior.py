# -*- coding: utf-8 -*-
import torch

from .constants import *


def ln_pdf(h,name,args):
    if name == "normal":
        mu,sigma2 = args
        return -(h - mu)**2/(2*sigma2) - 0.5*np.log(2*PI*sigma2)
    elif name == "lognormal":
        mu,sigma2 = args
        return -(torch.log(h) - mu)**2/(2*sigma2) - 0.5*np.log(2*PI*sigma2) - \
                 torch.log(h)
    else:
        raise ValueError
