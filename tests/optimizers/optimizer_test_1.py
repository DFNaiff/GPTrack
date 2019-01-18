# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

from src import optimizer

x = torch.ones(3,requires_grad=True)
def f(x):
    return torch.sum((x+1)**2)
opt = optimizer.lbfgs.LBFGS([x],
                      line_search_fn = "backtracking")

def closure():
    opt.zero_grad()
    y = f(x)
    y.backward(retain_graph=True)
    return y
    
y = opt.step(closure)
print(x,y,0)
    