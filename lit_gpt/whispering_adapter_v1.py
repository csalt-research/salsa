# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Port for Lit-GPT
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
from typing_extensions import Self

from lit_gpt.config import Config as BaseConfig
from lit_gpt.model import GPT as BaseModel
from lit_gpt.model import Block as BaseBlock
from lit_gpt.model import CausalSelfAttention


@dataclass
class Config(BaseConfig):
    adapter_start_layer: int = 1
    no_whisper_decoder_layers: int = 4
    no_layers_per_adapter: int = 8
    whisper_dim: int = 384
    downsampling_factor: int = 1
    normalize_before: bool = False

class BottleneckSiLU(nn.Module):
    '''
        This class is directly adapted from `lit_gpt.model.LLaMAMLP` class.
    '''
    def __init__(self, input_dim, output_dim, downsampling_factor) -> None:
        super().__init__()

        hidden_dim = input_dim // downsampling_factor
        self.fc_1 = nn.Linear(input_dim, hidden_dim, bias=False) # 1024 -> 256
        self.fc_2 = nn.Linear(input_dim, hidden_dim, bias=False) # 1024 -> 256
        self.proj = nn.Linear(hidden_dim, output_dim, bias=False) # 256 -> 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)

class GPT(BaseModel):
    """The implementation is identical to `lit_gpt.model.GPT` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None
        
        if self.config.downsampling_factor == 1:
            # Use simple projection
            self.whisper_proj = nn.ModuleList(
                [nn.Linear(config.whisper_dim, config.n_embd) for _ in range(config.no_layers_per_adapter)]
            )
        else:
            # Use down projection with SiLU activation like LLama models
            self.whisper_proj = nn.ModuleList(
                [BottleneckSiLU(config.whisper_dim, config.n_embd, self.config.downsampling_factor) for _ in range(config.no_layers_per_adapter)]
            )
        
        self.whisper_norm = None
        if self.config.normalize_before:
            print('Normalize done before')
            self.whisper_norm = config.norm_class(config.whisper_dim, eps=config.norm_eps)
        
        self.no_llama_decoders_per_adapter = math.ceil(config.n_layer / config.no_layers_per_adapter)
        self.no_whisper_decoders_per_adapter = math.ceil(config.no_whisper_decoder_layers / config.no_layers_per_adapter)
        
        self.index_cache = torch.arange(self.config.block_size).unsqueeze(-1).repeat(1,self.config.block_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, 
        idx: torch.Tensor, 
        whisper_states: torch.Tensor, 
        whisper_states_mapping: torch.Tensor, 
        input_pos: Optional[torch.Tensor] = None, 
        lm_head_chunk_size: int = 0
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        B = idx.size(0)
        T = idx.size(1)
        
        assert len(whisper_states.shape) == 4, f'Whisper state size does not have four dimensions'
        whisper_states = whisper_states[:,::self.no_whisper_decoders_per_adapter]
        if self.whisper_norm is not None:
            whisper_states = self.whisper_norm(whisper_states)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        # (B, N, T, D) -> (B, T, N, D)
        if whisper_states_mapping is not None:
            whisper_states_valid = whisper_states.transpose(1,2)[self.index_cache[:B,:T], whisper_states_mapping] # (B, T, N, D)
            whisper_states_valid = self.dropout(whisper_states_valid)
        else:
            # Activated during inference. Since we send only one 
            whisper_states_valid = whisper_states[:,:,-1].repeat(1,1,1,1) # Add time axis.
            
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for bi, block in enumerate(self.transformer.h):
            x = block(x, cos, sin, mask, input_pos)
            if bi % self.no_llama_decoders_per_adapter == 0:
                ind = bi // self.no_llama_decoders_per_adapter
                ws = whisper_states_valid[:,:,ind]
                x = x + self.whisper_proj[ind](ws)
        x = self.transformer.ln_f(x)
        if lm_head_chunk_size > 0:
            # chunk the lm head logits to reduce the peak memory used by autograd
            return [self.lm_head(x_i) for x_i in x.split(lm_head_chunk_size, dim=1)]
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, CausalSelfAttention):
            module.reset_parameters()

class Block(BaseBlock):
    """The implementation is identical to `lit_gpt.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: Config, block_idx: int) -> None:
        # Skip the parent class __init__ altogether and replace it to avoid useless allocations
        nn.Module.__init__(self)
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config


def mark_only_adapter_as_trainable(model: GPT) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = adapter_filter(name, param)


def adapter_filter(key: str, value: Any) -> bool:
    return "whisper" in key 
