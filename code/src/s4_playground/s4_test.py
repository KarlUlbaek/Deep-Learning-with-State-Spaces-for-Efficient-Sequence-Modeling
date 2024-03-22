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

from mamba_fork.mamba_ssm.modules.mamba_simple import S6MambaModule as Mamba_fast
from mamba_fork.mamba_ssm.modules.mamba_simple import MambaBlock as Mamba_fastBlock

import time
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
### install instructions
# cuda 11.7
# python 3.10.11
# nvidia driver 535
# gcc, make, g++
# torch 1.13.1 with cuda 117
# torchaudio          0.13.1
# torchmetrics        1.3.2
# torchvision         0.14.1
# pytorch_lightning, scipy (unversioned)
#

n_layers = 10
d_data = 32
d_model = 1024
d_state = 16
L = 512
B = 16
d = "cuda"
#print(torch.cuda.is_available())

# git test
modelargs = ModelArgs(d_model=d_model, n_layer=1, vocab_size=d_data, discrete_vocab=False)
mamba_slow = Mamba_slow(modelargs).to("cuda")
from s4_fork.models.s4.s4 import S4Block, FFTConvLean, FFTConv

# s4d = S4Block(d_model, mode="diag", init="legs", transposed=False).to(d)
#4d = S4D(model_D, init="legs", transposed=False).to(d)
# s4 = S4Block(d_model, mode="dplr", init="legs", transposed=False).to(d)
#s4_lean = S4BlockLean(model_D, mode="dplr", init="legs", transposed=False).to(d)
# mamba_fast = Mamba_fast(d_model, d_state).to(d)
# mamba_fast_block = Mamba_fastBlock(d_model, d_state, mixer_cls=Mamba_fast, fused_add_norm=False).to(d)


batch = torch.randn(B, L, d_data).to(d)

from mamba_fork.mamba_ssm.models.mixer_seq_simple import MixerModel as MambaNN
s6NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, discrete=False, fused_add_norm=False).to(d)

s4NN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, discrete=False, fused_add_norm=False,
               s4={"mode":"dplr", "hippo_init":"legs"}).to(d)
s4dNN = MambaNN(n_layer=n_layers, d_model=d_model, vocab_size=d_data, d_state=d_state, discrete=False, fused_add_norm=False,
               s4={"mode":"diag", "hippo_init":"legs"}).to(d)

#s4conv = FFTConvLean(d_model=d_model, d_state=32, mode="dplr", transposed=False, init="legs").to(d)

#s4conv(batch)
#print(s4conv)
#mamba_fast(batch)
#mamba_fast_block(batch)
reps = 10
for model in [s4NN, s4dNN, s6NN]:
   print("name:", model.layers[0].mixer.__class__.__name__)
   print("trainable params:", sum([param.numel() for param in model.parameters() if param.requires_grad]))

   for _ in range(2):
      model(batch)
   torch.cuda.synchronize()

   torch.cuda.reset_peak_memory_stats()
   t0 = time.perf_counter()
   for _ in range(reps):
      model(batch)
   torch.cuda.synchronize()
   mem1 = torch.cuda.max_memory_allocated()
   print("farward throughput: (s per batch)", (time.perf_counter()-t0)/reps)

   torch.cuda.reset_peak_memory_stats()
   t0 = time.perf_counter()
   for _ in range(int(reps*0.5)):
      (model(batch) - batch).sum().backward()
   torch.cuda.synchronize()
   print("back and farward throughput (s per batch):", (time.perf_counter()-t0)/(reps*0.5))
   print("peak mem: (farward, backward):", mem1/1e9, torch.cuda.max_memory_allocated()/1e9)

   #print(model)
   print("###################\n")
#mamba_slow(batch)
# s4d(batch)
# s4(batch)
#s4_lean(batch)

#print(s4_lean)

# import os
#
# print(os.getcwd())
# print(os.listdir(os.getcwd()))








