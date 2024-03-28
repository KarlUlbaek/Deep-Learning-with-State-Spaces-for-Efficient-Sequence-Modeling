import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
contract = torch.einsum
from causal_conv1d import causal_conv1d_fn
import sys
import os
sys.path.append(os.getcwd())

from mamba_ssm.ops.triton.layernorm import RMSNorm
from mamba_fork.mamba_ssm.modules.mamba_simple import S6MambaModule

from mamba_ssm import selective_scan_fn
from s4_fork.models.s4.s4 import DropoutNd, SSMKernelDiag, SSMKernelDPLR
kernel_registry = {
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
    's4': SSMKernelDPLR,
    'nplr': SSMKernelDPLR,
    'dplr': SSMKernelDPLR,
}

class FFTConvLean(nn.Module):
   """Implements an FFT Convolution around a convolution kernel.

   d_model (H): Model dimension (in CNN terminology, this would be "channels").
   l_max (L): The maximum kernel length. Set l_max=None to always use a global kernel.
   channels: Can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this; instead, increase d_model for larger models.
   bidirectional: If True, convolution kernel will be two-sided.
   activation: Activation after the full convolution.
   transposed, dropout, tie_dropout: More general model options, see SequenceModule.
   mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D. Other options can be found in the kernel registry.

   kernel_args: See the class .kernel.SSMKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode", "init", "dt_min", "dt_max", "lr"
   """

   def __init__(
      self,
      d_model,
      l_max=None,
      channels=1,
      transposed=True,
      dropout=0.0,
      tie_dropout=False,
      #drop_kernel=0.0,
      mode='dplr',
      #kernel=None,
      **kernel_args,  # Arguments passed into inner convolution kernel
   ):
      super().__init__()
      self.d_model = d_model
      self.L = self.l_max = l_max
      self.channels = channels
      self.BDL_shape = transposed

      self.D = nn.Parameter(torch.randn(channels, self.d_model))
      self.D._optim = True  # will get lower learning rate
      self.D._no_weight_decay = True  # will not get weight decaay

      kernel_cls = kernel_registry[mode]
      self.kernel = kernel_cls(
         d_model=self.d_model,
         l_max=self.l_max,
         channels=channels,
         **kernel_args,
      )

      dropout_fn = DropoutNd if tie_dropout else nn.Dropout
      self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
      #self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

   def forward(self, x, state=None, rate=1.0, **kwargs):  # absorbs return_output and transformer src mask
      """
      x: (B D L) if self.transposed else (B L D)
      """
      # Always work with (B D L) dimension in this module
      if not self.BDL_shape: x = x.transpose(-1, -2)
      L = x.size(-1)
      # Compute SS Kernel
      l_kernel = L# if self.L is None else min(L, round(self.L / rate))
      k, k_state = self.kernel(L=l_kernel, rate=rate, state=state)  # (C H L) (B C H L)
      #k = self.drop_kernel(k)

      # In principle, we could pad to l_kernel+L-1 instead of l_kernel+L, but we choose the latter for
      # equational simplicity. Additionally, we have not experimented to compare the efficiency of the two.
      k_f = torch.fft.rfft(k, n=l_kernel + L)  # (C H L)
      x_f = torch.fft.rfft(x, n=l_kernel + L)  # (B H L)
      y_f = contract('bhl,chl->bchl', x_f, k_f)
      y = torch.fft.irfft(y_f, n=l_kernel + L)[..., :L]  # (B C H L)

      # Compute D term in state space equation - essentially a skip connection
      y = y + contract('bhl,ch->bchl', x, self.D)

      y = rearrange(y, 'b c h l -> b (c h) l')

      y = self.drop(y)  # DropoutNd better with transposed=True

      if not self.BDL_shape: y = y.transpose(-1, -2)

      return y


class s4MambaModule(nn.Module):
   def __init__(
      self,
      d_model,
      d_state=16,
      dropout=0.0,
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

      self.s4fft = FFTConvLean(self.d_inner, d_state=d_state, mode=mode, init=hippo_init)

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

      self.dropout = DropoutNd(p=dropout) if dropout > 0.0 else nn.Identity()
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

      xz = self.dropout(xz)
      x, z = xz.chunk(2, dim=1)

      x = causal_conv1d_fn(
         x=x,
         weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
         bias=self.conv1d.bias,
         activation=self.activation,
      )

      x = self.s4fft(x)

      x = x * self.act(z)

      x = self.dropout(x)
      x = rearrange(x, "b d l -> b l d")
      out = self.out_proj(x)

      return out
class s4ClassicModule(nn.Module):
   def __init__(
      self,
      d_model,
      d_state=64,
      dropout=0.0,
      layer_idx=None,
      mode="diag",
      hippo_init ="legs"
   ):
      #factory_kwargs = {"device": device, "dtype": dtype}
      super().__init__()

      self.d_model = d_model
      self.d_state = d_state
      self.layer_idx = layer_idx
      self.s4fft = FFTConvLean(d_model, d_state=d_state, mode=mode, init=hippo_init)

      self.activation = nn.GELU()
      # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
      self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

      # position-wise output transform to mix features
      self.output_linear = nn.Sequential(
         nn.Conv1d(self.d_model, 2 * self.d_model, kernel_size=1),
         nn.GLU(dim=-2),
      )


   def forward(self, hidden_states, inference_params=None):
      #batch, seqlen, dim = hidden_states.shape
      #x = rearrange(hidden_states, "b d l -> b l d")
      x = self.s4fft(hidden_states)

      x = self.dropout(self.activation(x))

      x = self.output_linear(x)

      #x = rearrange(x, "b l d -> b d l")


      return x


class s6ClassicModule(nn.Module):
   def __init__(
      self,
      d_model,
      dropout=0.0,
      mode=None,
      d_state=16,
      dt_rank="auto",
      dt_min=0.001,
      dt_max=0.1,
      dt_init="random",
      dt_scale=1.0,
      dt_init_floor=1e-4,
      layer_idx=None,
      device=None,
      dtype=None,
   ):
      factory_kwargs = {"device": device, "dtype": dtype}
      super().__init__()
      self.d_model = d_model
      self.d_state = d_state
      self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
      self.layer_idx = layer_idx

      #self.activation = "silu"
      #self.act = nn.SiLU()

      self.x_proj = nn.Linear(
         self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
      )
      self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True, **factory_kwargs)

      # Initialize special dt projection to preserve variance at initialization
      dt_init_std = self.dt_rank ** -0.5 * dt_scale
      if dt_init == "constant":
         nn.init.constant_(self.dt_proj.weight, dt_init_std)
      elif dt_init == "random":
         nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
      else:
         raise NotImplementedError

      # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
      dt = torch.exp(
         torch.rand(self.d_model, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
         + math.log(dt_min)
      ).clamp(min=dt_init_floor)
      # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
      inv_dt = dt + torch.log(-torch.expm1(-dt))
      with torch.no_grad():
         self.dt_proj.bias.copy_(inv_dt)
      # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
      self.dt_proj.bias._no_reinit = True

      # S4D real initialization
      A = repeat(
         torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
         "n -> d n",
         d=self.d_model,
      ).contiguous()
      A_log = torch.log(A)  # Keep A_log in fp32
      self.A_log = nn.Parameter(A_log)
      self.A_log._optim = True
      self.A_log._no_weight_decay = True

      # D "skip" parameter
      self.D = nn.Parameter(torch.ones(self.d_model, device=device))  # Keep in fp32
      self.D._optim = True
      self.D._no_weight_decay = True


      self.activation = nn.GELU()
      self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

      # position-wise output transform to mix features
      self.output_linear = nn.Sequential(
         nn.Conv1d(self.d_model, 2 * self.d_model, kernel_size=1),
         nn.GLU(dim=-2),
      )

   def forward(self, hidden_states, inference_params=None):
      """
      hidden_states: (B, D, L)
      Returns: same shape as hidden_states
      """
      batch, dim, seqlen = hidden_states.shape
      conv_state, ssm_state = None, None

      A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
      x = hidden_states #x = b d l
      z = x.clone()#torch.ones_like(x)*1.2785 # silu(torch.ones(1,)*1.2785)=1 such that the skip connection doesnt do anything
      x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
      dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
      dt = self.dt_proj.weight @ dt.t()
      dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
      B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
      C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
      #assert self.activation in ["silu", "swish"]
      y = selective_scan_fn(
         x,
         dt,
         A,
         B,
         C,
         self.D.float(),
         z=z,
         delta_bias=self.dt_proj.bias.float(),
         delta_softplus=True,
         return_last_state=ssm_state is not None,
      )

      y = self.dropout(self.activation(y))

      y = self.output_linear(y)

      return y


class S4Model(nn.Module):
   def __init__(
      self,
      d_input,
      d_state=64,
      s4_or_s6 = s4ClassicModule,
      d_output=10,
      d_model=128,
      n_layers=4,
      dropout=0.2,
      prenorm=False,
      layernorm=True,
      mode = "diag",
   ):
      super().__init__()

      self.prenorm = prenorm
      self.encoder = nn.Linear(d_input, d_model)
      self.NotMambaShape = True if s4_or_s6.__name__ != "S6MambaModule" else False

      # Stack S4 layers as residual blocks
      self.s4_layers = nn.ModuleList()
      self.norms = nn.ModuleList()
      self.dropouts = nn.ModuleList()
      norm_fn = nn.LayerNorm if layernorm else RMSNorm
      for _ in range(n_layers):
         self.s4_layers.append(s4_or_s6(d_model, d_state=d_state, dropout=dropout, mode=mode))
         self.norms.append(norm_fn(d_model))
         self.dropouts.append(nn.Dropout1d(dropout))

      # Linear decoder
      self.decoder = nn.Linear(d_model, d_output)

   def forward(self, x):
      """
      Input x is shape (B, L, d_input)
      """
      x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

      #print(self.NotMambaShape)
      if self.NotMambaShape: x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
      for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
         # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

         residuals = x
         if self.prenorm:
            if self.NotMambaShape: norm(x.transpose(-1, -2)).transpose(-1, -2)
            else:                  norm(x)

         # Apply S4 block: we ignore the state input and output
         x = layer(x)
         x = dropout(x)
         x = x + residuals

         if not self.prenorm:
            if self.NotMambaShape: norm(x.transpose(-1, -2)).transpose(-1, -2)
            else:                  norm(x)

      if self.NotMambaShape: x = x.transpose(-1, -2)

      # Pooling: average pooling over the sequence length
      x = self.decoder(x)  # (B, d_model) -> (B, d_output)
      x = x.mean(dim=1)

      return x

if __name__ == "__main__":
   d_input = 64
   d_model = 128
   d_state = 64
   L = 512
   B = 16
   d = "cuda"

   m = S6MambaModule(64, 64)



   s4model = S4Model(
                     d_input = d_input,
                     d_state = d_state,
                     s4_or_s6=S6MambaModule,
                     d_output=d_input,
                     d_model=d_model,
                     n_layers=4,
                     dropout=0.1,
                     prenorm=False,
                     mode="diag",
                  ).to(d)

   batch = torch.randn(B, L, d_input).to(d)

   s4model(batch)
   print(s4model)









