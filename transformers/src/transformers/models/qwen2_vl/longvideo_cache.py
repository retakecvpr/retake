import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, StaticCache, DynamicCache
from ...utils import logging


logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class PivotKVCache(DynamicCache):
    def __init__(self, config) -> None:
        super().__init__()
        # Patch longvideo kwargs
        kv_compression_kwargs = config.longvideo_kwargs['kvcache_compression_kwargs']
        self.kvcache_compression = True
        self.compression_ratio = kv_compression_kwargs['compression_ratio']
        self.compression_method = kv_compression_kwargs['compression_method']
        self.chunk_frames = kv_compression_kwargs['chunk_frames']

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            query_states: [bsz, num_heads, q_len, d]
            key_states: [bsz, num_key_value_heads, q_len, d]
        """
        logger.warning_once("Enable PivotKVCache compression: length after compression %.2f" % (self.compression_ratio))

        query_states = cache_kwargs.pop('query_states')
        # 1) Hidden states for the next layer remains uncompressed in current chunked prefill iter
        key_states_output, value_states_output = super().update(key_states, value_states, layer_idx, cache_kwargs)

        if self.kvcache_compression: # when prefilling visual tokens
            bsz, num_heads, q_len, head_dim = query_states.shape
            num_key_value_heads = key_states.shape[1]
            key_states_repeated = repeat_kv(key_states, num_heads // num_key_value_heads)
            assert bsz == 1

            # 2) Evit KV Cache based on query_states
            keep_len = int(self.compression_ratio * q_len) # Evict new tokens only
            attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # [bsz, self.num_heads, q_len, q_len(k)]
            attn_weights = attn_weights[0].sum(1) # [self.num_heads, q_len(k)]
            attn_weights = attn_weights.reshape(num_key_value_heads, -1, q_len).mean(1) # [num_key_value_heads, q_len(k)]
            attn_weights = attn_weights.mean(0, keepdim=True).repeat(num_key_value_heads, 1) # [num_key_value_heads, q_len(k)]

            if getattr(self, "keypatches_mask_chunk", None) is not None:
                keypatches_mask_chunk = self.keypatches_mask_chunk
                attn_weights.masked_fill_(keypatches_mask_chunk, 1.) # Select key patches first

            _, keep_indices = attn_weights.topk(keep_len, -1)
            keep_indices = keep_indices.sort().values # [self.num_key_heads, keep_len]
            keep_indices = keep_indices[None,:,:,None].repeat(bsz, 1, 1, head_dim) # [bsz, self.num_key_heads, keep_len, head_dim]

            compressed_key_states = torch.gather(input=key_states_output[...,-q_len:,:], dim=2, index=keep_indices)
            compressed_value_states = torch.gather(input=value_states_output[...,-q_len:,:], dim=2, index=keep_indices) # [bsz, num_k_heads, keep_len, head_dim]

            # 3) Update KVCache
            self.key_cache[layer_idx] = torch.cat([
                key_states_output[...,:-q_len,:], compressed_key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                value_states_output[...,:-q_len,:], compressed_value_states
            ], dim=2)
        else: # when prefilling textual tokens or decoding
            pass

        return key_states_output, value_states_output


def build_kvcache(config):
    if config.longvideo_kwargs is None or not config.longvideo_kwargs.get('kvcache_compression', False):
        return DynamicCache()
    else:
        compression_method = config.longvideo_kwargs['kvcache_compression_kwargs']['compression_method']
        if compression_method.lower() == 'pivotkv':
            return PivotKVCache(config)
        else:
            raise NotImplementedError