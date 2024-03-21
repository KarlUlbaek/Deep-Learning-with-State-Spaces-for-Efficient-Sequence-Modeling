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
#from s4_fork.models.s4.s4 import S4Block
from mamba_purepython_fork.mamba_slow_model import Mamba as Mamba_slow
from mamba_purepython_fork.mamba_slow_model import ModelArgs

from mamba_fork.mamba_ssm.modules.mamba_simple import Mamba as Mamba_fast

### install instructions
# cuda 11.7
# python 3.10.11
# nvidia driver 535
# torch 1.13.1 with cuda 117
# torchaudio          0.13.1
# torchmetrics        1.3.2
# torchvision         0.14.1
# pytorch_lightning, scipy (unversioned)
#


data_D = 64
model_D = data_D
L = 128
B = 16
d = "cuda"
print(torch.cuda.is_available())

# git test
modelargs = ModelArgs(d_model=model_D, n_layer=1, vocab_size=data_D, discrete_vocab=False)
mamba_slow = Mamba_slow(modelargs).to("cuda")
from s4_fork.models.s4.s4 import S4Block
s4d = S4Block(model_D, mode="diag", init="legs", transposed=False).to(d)
#4d = S4D(model_D, init="legs", transposed=False).to(d)
s4 = S4Block(model_D, mode="dplr", init="legs", transposed=False).to(d)
mamba_fast = Mamba_fast(d_model=data_D).to(d)


batch = torch.randn(B, L, data_D).to(d)

mamba_fast(batch)
#mamba_slow(batch)
s4d(batch)
s4(batch)

# import os
#
# print(os.getcwd())
# print(os.listdir(os.getcwd()))








