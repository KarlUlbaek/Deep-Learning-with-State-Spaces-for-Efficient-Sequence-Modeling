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

from s4_playground.misc import RotaryEmbeddingCustom
from mamba_ssm.ops.triton.layernorm import RMSNorm
from mamba_fork.mamba_ssm.modules.mamba_simple import S6MambaModule

from mamba_ssm import selective_scan_fn
from s4_fork.models.s4.s4 import SSMKernelDiag, SSMKernelDPLR
kernel_registry = {
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
    's4': SSMKernelDPLR,
    'nplr': SSMKernelDPLR,
    'dplr': SSMKernelDPLR,
}

class unidirectional(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "unidirectional"
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.kernel = kernel_cls(d_model=d_model, l_max=l_max,
                               channels=channels, init=init, d_state=d_state)

   def forward(self, x):
      L = x.size(-1)
      k, _ = self.kernel(L=L)  # (H L)

      k_f = torch.fft.rfft(k, n=2 * L)  # (1 C L)
      x_f = torch.fft.rfft(x, n=2 * L)  # (B C L)
      y = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]  # (B H L)
      y += x * self.D
      return y

class unidirectional_placebo(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "unidirectional_placebo"
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.k1 = kernel_cls(d_model=d_model, l_max=l_max,
                                  channels=channels, init=init, d_state=d_state)
      self.k2 = kernel_cls(d_model=d_model, l_max=l_max,
                                  channels=channels, init=init, d_state=d_state)
   def forward(self, x):
      b, c, L = x.shape
      k1, _ = self.k1(L=L)
      k_f = torch.fft.rfft(k1, n=2 * L)  # (1, H L)
      x_f = torch.fft.rfft(x, n=2 * L)  # (B H L)
      y1 = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]

      k2, _ = self.k2(L=L)
      k_f = torch.fft.rfft(k2, n=2 * L)  # (1, H L)
      y2 = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]

      # if self.use_feature_mix:
      #    y = self.feature_mixer(torch.cat([y1, y2.flip(-1)], dim=1))
      # else:
      y = y1 + y2

      y += x * self.D
      return y

class sequential_bi_tied(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "sequential_bi_tied"
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.kernel = kernel_cls(d_model=d_model, l_max=l_max,
                               channels=channels, init=init, d_state=d_state)

   def forward(self,x):
      b, c, L = x.shape
      k, _ = self.kernel(L=L)
      k_f = torch.fft.rfft(k, n=2 * L)  # (1, H L)

      x1 = torch.fft.rfft(x, n=2 * L)  # (B H L)
      x2 = torch.fft.rfft(x.flip(-1), n=2 * L)  # (B H L)

      y1 = torch.fft.irfft(x1 * k_f, n=2 * L)[..., :L]
      y2 = torch.fft.irfft(x2 * k_f, n=2 * L)[..., :L]
      #
      # if self.use_feature_mix:
      #    y = self.feature_mixer(torch.cat([y1, y2.flip(-1)], dim=1))
      # else:
      y = y1 + y2.flip(-1)

      y += x * self.D
      return y

class half_dim_bi(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "half_dim_bi"
      self.halv_dim = int(d_model / 2)
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.kernel = kernel_cls(d_model=d_model, l_max=l_max,
                               channels=channels, init=init, d_state=d_state)

   def forward(self, x):
      L = x.size(-1)
      k, _ = self.kernel(L=L)  # (H L)
      res = x.clone() # this one modifies x in place so we need to make a copy

      self.halv_dim = int(self.d_model/2)
      x[:, :self.halv_dim, :] = x[:, :self.halv_dim, :].flip(-1)
      k_f = torch.fft.rfft(k, n=2 * L)  # (1 C L)
      x_f = torch.fft.rfft(x, n=2 * L)  # (B C L)
      y = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]  # (B H L)

      y[:, :self.halv_dim, :] = y[:, :self.halv_dim, :].flip(-1)

      y += res * self.D
      return y

class paper_bi(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "paper_bi"
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.kernel = kernel_cls(d_model=d_model, l_max=l_max,
                               channels=channels*2, init=init, d_state=d_state)
   def forward(self, x):
      L = x.size(-1)
      k, _ = self.kernel(L=L)  # (H L)

      k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
      k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

      k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
      x_f = torch.fft.rfft(x, n=2 * L)  # (B H L)
      y_f = torch.einsum('bhl,chl->bchl', x_f, k_f)
      y = (torch.fft.irfft(y_f, n=2 * L)[..., :L]).squeeze()  # (B C H L)
      y += x * self.D
      return y

class stacked_bi(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "stacked_bi"
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.kernel = kernel_cls(d_model=d_model*2, l_max=l_max,
                               channels=channels, init=init, d_state=d_state)
   def forward(self, x):
      b, c, L = x.shape
      k, _ = self.kernel(L=L)

      x_new = torch.cat([x, x.flip(-1)], dim=-2)
      k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
      x_f = torch.fft.rfft(x_new, n=2 * L)  # (B H L)
      y = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]
      # if self.use_feature_mix:
      #    y[:, c:, :] = y[:, c:, :].flip(-1) # might not work
      #    y = self.feature_mixer(y)
      #else:
      y = y[:, :c, :] + y[:, c:, :].flip(-1)

      y += x * self.D
      return y

class sequential_bi(nn.Module):
   def __init__(self, kernel_cls, d_model, d_state, channels, l_max, init):
      super().__init__()
      self.name = "sequential_bi"
      self.D = nn.Parameter(torch.ones((channels, d_model, 1), dtype=torch.float))
      self.k1 = kernel_cls(d_model=d_model, l_max=l_max,
                                  channels=channels, init=init, d_state=d_state)
      self.k2 = kernel_cls(d_model=d_model, l_max=l_max,
                                  channels=channels, init=init, d_state=d_state)
   def forward(self, x):
      b, c, L = x.shape
      k1, _ = self.k1(L=L)
      k_f = torch.fft.rfft(k1, n=2 * L)  # (1, H L)
      x_f = torch.fft.rfft(x, n=2 * L)  # (B H L)
      y1 = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]

      k2, _ = self.k2(L=L)
      k_f = torch.fft.rfft(k2, n=2 * L)  # (1, H L)
      x_f = torch.fft.rfft(x.flip(-1), n=2 * L)  # (B H L)
      y2 = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]

      # if self.use_feature_mix:
      #    y = self.feature_mixer(torch.cat([y1, y2.flip(-1)], dim=1))
      # else:
      y = y1 + y2.flip(-1)

      y += x * self.D
      return y

direction_registry = {
      "paper_bi":paper_bi,
      "stacked_bi":stacked_bi,
      "sequential_bi":sequential_bi,
      "sequential_bi_tied":sequential_bi_tied,
      "half_dim_bi":half_dim_bi,
      "placebo": unidirectional_placebo,
      "": unidirectional
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
      mode='dplr',
      init='legs',
      bi = "",
      #use_feature_mix = "",
      d_state=64,  # Arguments passed into inner convolution kernel
   ):
      super().__init__()
      self.d_model = d_model
      self.L = self.l_max = l_max
      self.channels = channels
      self.BDL_shape = transposed

      assert bi in ["paper_bi", "stacked_bi", "sequential_bi", "sequential_bi_tied", "half_dim_bi", "", "placebo"]
      #if bi != "": print(f"using {bi} kernel")
      self.bi = bi
      #self.use_feature_mix = use_feature_mix

      kernel_cls = kernel_registry[mode]
      direction_cls = direction_registry[bi]

      # (self, kernel_cls, d_model, d_state, channels, l_max, init):
      self.fftconv = direction_cls(kernel_cls=kernel_cls, d_model=d_model, d_state=d_state, channels=channels,
                                   l_max=l_max, init=init)

   def forward(self, x):
      return self.fftconv(x)


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
      s4_kwargs = {},
      pos_emb = {}
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

      self.s4fft = FFTConvLean(self.d_inner, d_state=d_state, **s4_kwargs)

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

      self.dropout = nn.Dropout1d(p=dropout) if dropout > 0.0 else nn.Identity()
      self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

      self.use_pos_emb = bool(pos_emb)
      if self.use_pos_emb:
         if self.layer_idx == 0: print("using pos {} embddings".format(pos_emb["loc"]))
         self.pos_emb_layer = RotaryEmbeddingCustom(d_model=self.d_inner, **pos_emb, BDL_shape=True)

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

      if self.use_pos_emb:
         x = self.pos_emb_layer(x, layer_idx=self.layer_idx)

      x = self.s4fft(x)
      x = x * self.act(z)

      #x = self.dropout(x)
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
      s4_kwargs = {"mode":"diag", "init":"legs"},
      pos_emb = {}
   ):
      #factory_kwargs = {"device": device, "dtype": dtype}
      super().__init__()

      self.d_model = d_model
      self.d_state = d_state
      self.layer_idx = layer_idx
      self.s4fft = FFTConvLean(d_model, d_state=d_state, **s4_kwargs)

      self.activation = nn.GELU()
      # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
      self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

      # position-wise output transform to mix features
      self.output_linear = nn.Sequential(
         nn.Conv1d(self.d_model, 2 * self.d_model, kernel_size=1),
         nn.GLU(dim=-2),
      )
      self.use_pos_emb = bool(pos_emb)
      if self.use_pos_emb:
         if self.layer_idx == 0: print("using pos {}".format(pos_emb["loc"]))
         self.pos_emb_layer = RotaryEmbeddingCustom(d_model=self.d_model, **pos_emb, BDL_shape=True)

   def forward(self, hidden_states, inference_params=None):
      #batch, seqlen, dim = hidden_states.shape
      #x = rearrange(hidden_states, "b d l -> b l d")
      if self.use_pos_emb:
         hidden_states = self.pos_emb_layer(hidden_states, self.layer_idx)

      x = self.s4fft(hidden_states)


      x = self.dropout(x)

      x = self.activation(x)

      x = self.output_linear(x)

      #x = rearrange(x, "b l d -> b d l")


      return x



class S4ClassicModel(nn.Module):
   def __init__(
      self,
      d_input: int,
      d_output: int,
      classification=True,
      vocab_size = None, # implies a discrete input
      d_model=128,
      d_state=64,
      n_layer=4,
      dropout=0.0,
      prenorm=False,
      layernorm=True,
      s4_kwargs = {"mode": "dplr", "init": "legs"},
      pos_emb = {},
      bi_module = {},
      reversed_pre=False
   ):
      super().__init__()
      self.d_model, self.d_state, self.n_layer, self.dropout, self.s4 = d_model, d_state, n_layer, dropout, s4_kwargs["mode"]
      self.d_input, self.d_output, self.vocab_size = d_input, d_output, vocab_size
      self.prenorm = prenorm
      self.s4_kwargs = s4_kwargs
      self.pos_emb = pos_emb
      self.bi_module = bi_module
      self.reversed_pre= reversed_pre

      if vocab_size is not None:
         self.encoder = nn.Embedding(vocab_size, d_model)
      else:
         self.encoder = nn.Linear(d_input, d_model)

      self.classification = classification

      # Stack S4 layers as residual blocks
      self.layers = nn.ModuleList()
      self.norms = nn.ModuleList()
      self.dropouts = nn.ModuleList()
      norm_fn = nn.LayerNorm if layernorm else RMSNorm
      for layer_idx in range(n_layer):
         self.layers.append(s4ClassicModule(d_model, d_state=d_state, layer_idx=layer_idx,
                                            dropout=dropout, pos_emb=pos_emb, s4_kwargs=s4_kwargs))
         self.norms.append(norm_fn(d_model))
         self.dropouts.append(nn.Dropout1d(dropout))

      # Linear decoder
      if self.classification:
         self.decoder = nn.Linear(d_model, d_output)
      else:
         self.decoder = nn.Linear(d_model, vocab_size)
         #self.tie_weights()

   # def tie_weights(self):
   #    self.decoder.weight = self.encoder.weight

   def forward(self, x):
      """
      Input x is shape (B, L, d_input)
      """
      x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

      #print(self.NotMambaShape)
      x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
      for layer_idx, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
         # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

         res = x
         if self.prenorm:
            res = norm(res.transpose(-1, -2)).transpose(-1, -2)

         res = layer(res)
         res = dropout(res)
         x = x + res

         if not self.prenorm:
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)

      x = x.transpose(-1, -2)

      # Pooling: average pooling over the sequence length
      if self.classification:
         x = x.mean(dim=1)

      hidden_states = self.decoder(x)
      return hidden_states

if __name__ == "__main__":
   from LRA_baseline import model_throughput
   d_input = 64
   d_model = 128*2
   d_state = 64
   L = 512*2
   B = 64
   d = "cuda"
   d_output = 10

   s4model = S4ClassicModel(
                           d_input = d_input,
                           d_output = d_output,
                           pos_emb=False
                           ).to(d)

   batch = torch.randn(B, L, d_input).to(d)



   s4model(batch)


   model_throughput(s4model, None, d_input, L=L, b=B)

   print("with pos")
   s4model = S4ClassicModel(
                           d_input = d_input,
                           d_output = d_output,
                           pos_emb=True
                           ).to(d)

   batch = torch.randn(B, L, d_input).to(d)

   s4model(batch)
   model_throughput(s4model, None, d_input, L=L, b=B)


   #print(s4model)









