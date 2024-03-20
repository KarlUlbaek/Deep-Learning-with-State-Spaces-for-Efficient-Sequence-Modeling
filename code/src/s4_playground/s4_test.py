# import torch
# import torchvision
# #from torch.utils import DataLoader
#
# trainset = torchvision.datasets.CIFAR10(train=True, download=True)
# testset = torchvision.datasets.CIFAR10(train=False, download=True)
#
# class cifar10data(torch.utils.Dataset):
import torch

from s4_fork.models.s4.s4d import S4D
from s4_fork.models.s4.s4 import S4Block
from mamba_purepython_fork.mamba_slow_model import Mamba as Mamba_slow
from mamba_purepython_fork.mamba_slow_model import ModelArgs


data_D = 64
model_D = data_D
L = 128
B = 16



# git test
modelargs = ModelArgs(d_model=model_D, n_layer=1, vocab_size=data_D, discrete_vocab=False)
mamba_slow = Mamba_slow(modelargs)
s4d = S4Block(model_D, mode="diag", init="legs", transposed=False)
s4 = S4Block(model_D, mode="dplr", init="legs", transposed=False)



b = torch.randn(B, L, data_D) #

mamba_slow(b)
s4d(b)
s4(b)

# import os
#
# print(os.getcwd())
# print(os.listdir(os.getcwd()))








