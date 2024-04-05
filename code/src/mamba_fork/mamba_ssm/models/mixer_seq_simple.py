# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import S6MambaModule, S6MambaModulePosEmb ,MambaBlock
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
import os
from s4_playground.s4_modules import s4MambaModule
from s4_playground.misc import RotaryEmbeddingCustom

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class BiModule(nn.Module):
   def __init__(self, partial_model, d_model, d_state,
                d_model_scale, d_state_scale, placebo = False):
      super().__init__()
      self.placebo = placebo
      self.forward_model = partial_model(d_model=int(d_model*d_model_scale),
                                         n_state=int(d_state * d_state_scale))

      self.backward_model = partial_model(d_model=int(d_model*d_model_scale),
                                          n_state=int(d_state * d_state_scale))

   def forward(self, x):
      if not self.placebo:
          forward = self.forward_model(x)
          backward = self.backward_model(x.flip(-2))

          return forward + backward.flip(-2)
      else:
          forward = self.forward_model(x)
          backward = self.backward_model(x)

          return forward + backward


def create_block(
    d_model,
    d_state,
    dropout=0.0,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    s4_kwargs ={}, # {mode:"dplr", hippo_init ="legs"}
    pos_emb={},
    bi_s6={},
    bi_module={}#{"d_model_scale": 0.66, "n_state_scale": 1.0}
):
    # if ssm_cfg is None:
    #     ssm_cfg = {}
    #
    # factory_kwargs = {"device": device, "dtype": dtype}
    if not bool(s4_kwargs):
        if not bool(pos_emb) and not bool(bi_s6):
            mixer_cls = partial(S6MambaModule, dropout=dropout, layer_idx=layer_idx)#, **ssm_cfg, **factory_kwargs)
        else:
            mixer_cls = partial(S6MambaModulePosEmb, dropout=dropout, layer_idx=layer_idx, pos_emb=pos_emb, bi=bi_s6)#, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(s4MambaModule, dropout=dropout, layer_idx=layer_idx, pos_emb=pos_emb, s4_kwargs=s4_kwargs)#, **ssm_cfg, **factory_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)#, **factory_kwargs
    #)
    if bi_module:
        assert not bi_s6, "s6 SSP is already birectional"
        assert ""==s4_kwargs.get("bi", ""), "s4 SSP is already {}".format(s4_kwargs.get("bi", 0))
        mixer_cls = BiModule(partial_model=mixer_cls, d_model=d_model, d_state=d_state,
                             **bi_module)
        norm_cls = norm_cls(d_model=int(d_model*bi_module["d_model_scale"]))
    else:
        mixer_cls = mixer_cls(d_model=d_model, d_state=d_state)
        norm_cls = norm_cls(d_model=d_model)


    block = MambaBlock(
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaModel(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        classification=True,
        vocab_size = None, # corrosponds to discrete input
        d_model= 128,
        d_state= 16,
        n_layer= 4,
        dropout= 0.0,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=True,
        s4_kwargs = {},  # {mode:"dplr", hippo_init ="legs"}
        pos_emb = {},
        bi_s6={}, # {"bi":True}
        bi_module = {}

    ) -> None:
        #factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.classification = classification
        self.d_model, self.d_state, self.n_layer, self.dropout = d_model, d_state, n_layer, dropout
        self.d_input, self.d_output, self.vocab_size= d_input, d_output, vocab_size
        self.s4 = "s6" if not bool(s4_kwargs) else s4_kwargs["mode"]
        self.s4_kwargs = s4_kwargs
        self.bi_s6 = bi_s6
        self.bi_module = bi_module

        if vocab_size:
            self.encoder = nn.Embedding(vocab_size, d_model)
        else:
            self.encoder = nn.Linear(d_input, d_model)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    s4_kwargs=s4_kwargs,  # {mode:"dplr", hippo_init ="legs"}
                    pos_emb=pos_emb,
                    bi_s6= bi_s6,
                    bi_module = bi_module
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, #**factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        if self.classification:
            self.decoder = nn.Linear(d_model, d_output)
        else:
            self.decoder = nn.Linear(d_model, vocab_size)
            # self.tie_weights()

        # def tie_weights(self):
        #    self.decoder.weight = self.encoder.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = self.encoder(input_ids)
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.classification:
            hidden_states = hidden_states.mean(dim=1)

        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MambaModel(
            d_model=d_model,
            n_layer=n_layer,
            d_input=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.encoder.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
