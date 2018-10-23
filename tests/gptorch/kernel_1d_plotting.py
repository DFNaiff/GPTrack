import sys
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt
import torch

from src import kernels,utilstorch

l1 = 1.0
T = 3.0
l = 1.0
hparams = [torch.tensor(l1),torch.tensor(T),torch.tensor(l)]
kernel = kernels.IsoMatern32(dim=1) + kernels.PerMatern32()
kernel.initialize(hparams)
Y = torch.linspace(0,10,201).reshape(-1,1)
X = torch.zeros_like(Y)
Z = kernel.f(X,Y)
plt.plot(Y.numpy(),Z.numpy(),'b')