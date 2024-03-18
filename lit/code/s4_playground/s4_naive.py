import torch
import torchvision
from torch.utils import DataLoader

trainset = torchvision.datasets.CIFAR10(train=True, download=True)
testset = torchvision.datasets.CIFAR10(train=False, download=True)

class cifar10data(torch.utils.Dataset):
   





