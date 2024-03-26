import torch
import torch.nn as nn
import torch.nn.functional as F
from s4_fork.models.s4.s4 import FFTConvLean
import math
from einops import rearrange, repeat
from causal_conv1d import causal_conv1d_fn


#FFTConvLean(d_model, d_state, mode="dplr", transposed=False, init="legs")

class s4MambaModule(nn.Module):
   def __init__(
      self,
      d_model,
      d_state=16,
      d_conv=4,
      expand=2,
      conv_bias=True,
      bias=False,
      use_fast_path=True,  # Fused kernel options
      layer_idx=None,
      device=None,
      dtype=None,
      mode="dplr",
      hippo_init ="legs"
   ):
      factory_kwargs = {"device": device, "dtype": dtype}
      super().__init__()
      self.d_model = d_model
      self.d_state = d_state
      self.d_conv = d_conv
      self.expand = expand
      self.d_inner = int(self.expand * self.d_model)
      self.layer_idx = layer_idx
      self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

      self.s4fft = FFTConvLean(self.d_inner, d_state=d_state, mode=mode, transposed=True, init=hippo_init)

      self.activation = "silu"
      self.act = nn.SiLU()

      self.conv1d = nn.Conv1d(
         in_channels=self.d_inner,
         out_channels=self.d_inner,
         bias=conv_bias,
         kernel_size=d_conv,
         groups=self.d_inner,
         padding=d_conv - 1,
         **factory_kwargs,
      )

      self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)


   def forward(self, hidden_states, inference_params=None):
      batch, seqlen, dim = hidden_states.shape

      # We do matmul and transpose BLH -> HBL at the same time
      xz = rearrange(
         self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
         "d (b l) -> b d l",
         l=seqlen,
      )
      if self.in_proj.bias is not None:
         xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

      x, z = xz.chunk(2, dim=1)

      x = causal_conv1d_fn(
         x=x,
         weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
         bias=self.conv1d.bias,
         activation=self.activation,
      )

      x = self.s4fft(x)

      x = x * self.act(z)
      x = rearrange(x, "b d l -> b l d")
      out = self.out_proj(x)

      return out

if __name__ == "__main__":
   d_data = 64
   d_model = d_data
   d_state = 32
   L = 512
   B = 16
   d = "cuda"

   batch = torch.randn(B, L, d_data).to(d)

   #ffttest = FFTConvLean(d_model, d_state=d_state, mode="dplr", transposed=False, init="legs").to(d)
   #ffttest(batch)

   s4mambastyle = s4MambaModule(d_model=d_model, d_state=d_state).to(d)
   print(s4mambastyle)
   s4mambastyle(batch)









