import torchvision

root = "../data/cifar10/"
trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

