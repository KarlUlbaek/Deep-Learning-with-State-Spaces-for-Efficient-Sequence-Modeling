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

s4 = S4Block(128, mode="diag", init="legs")






s4d = S4D(128, )



b = torch.randn(16,128,16)

s4d(b)
s4(b)

# import os
#
# print(os.getcwd())
# print(os.listdir(os.getcwd()))








