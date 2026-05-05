#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
# Vendored verbatim from
#   https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/multihead_attention.py
# at upstream HEAD as of 2026-05-05. See ./layer_norm.py for the rationale on
# vendoring rather than pip-depending on `ane-transformers`.
#

import torch
import torch.nn as nn

from .layer_norm import LayerNormANE


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention optimized for efficient ANE deployment"""

    def __init__(self, embed_dim, d_qk=None, d_v=None, d_out=None, n_head=8, dropout=0.1, **kwargs):
        super().__init__()

        self.d_qk = d_qk or embed_dim
        self.d_v = d_v or embed_dim
        self.d_out = d_out or embed_dim

        self.n_head = n_head
        if self.d_qk % self.n_head != 0 or self.d_v % self.n_head != 0:
            raise ValueError(
                f"Either query-key dimensions ({self.d_qk}) or the value embeddings "
                f"dimensions ({self.d_v}) is not divisible by n_head ({self.n_head})"
            )
        self.q_normalize_fact = float(self.d_qk // self.n_head) ** -0.5

        self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.v_proj = nn.Conv2d(embed_dim, self.d_v, 1)
        self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def _attention_fn(self, q, k, v, qk_mask, k_mask, return_weights):
        """Core routine for computing multi-head attention

        Shapes (BC1S layout):
            q: (batch_size, d_qk, 1, tgt_seq_len)
            k: (batch_size, d_qk, 1, src_seq_len)
            v: (batch_size, d_v,  1, src_seq_len)
            qk_mask: (batch_size, src_seq_len, 1, tgt_seq_len) — additive (e.g. -1e4 to mask)
            k_mask:  (batch_size, src_seq_len, 1, 1)           — additive (e.g. -1e4 to mask)
        """
        # Principle 2: Chunking Large Intermediate Tensors
        # Split q, k and v to compute a list of single-head attention functions
        mh_q = q.split(self.d_qk // self.n_head, dim=1)
        # Principle 3: Minimizing Memory Copies
        mh_k = k.transpose(1, 3).split(self.d_qk // self.n_head, dim=3)
        mh_v = v.split(self.d_v // self.n_head, dim=1)

        attn_weights = [torch.einsum("bchq,bkhc->bkhq", [qi, ki]) * self.q_normalize_fact for qi, ki in zip(mh_q, mh_k)]

        if qk_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + qk_mask
        if k_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + k_mask

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]
        mh_w = [self.dropout(aw) for aw in attn_weights]

        attn = [torch.einsum("bkhq,bchk->bchq", wi, vi) for wi, vi in zip(mh_w, mh_v)]
        attn = torch.cat(attn, dim=1)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def _forward_impl(
        self,
        q,
        k,
        v,
        qpos=None,
        kpos=None,
        vpos=None,
        qk_mask=None,
        k_mask=None,
        return_weights=False,
    ):
        assert len(q.size()) == 4 and len(k.size()) == 4 and len(v.size()) == 4
        b, ct, ht, wt = q.size()
        b, cs, hs, ws = k.size()

        tgt_seq_len = ht * wt
        src_seq_len = hs * ws

        if qpos is not None:
            q = q + qpos
        if kpos is not None:
            k = k + kpos
        if vpos is not None:
            v = v + kpos

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        expected_qk_mask_shape = [b, src_seq_len, 1, tgt_seq_len]
        if qk_mask is not None:
            if qk_mask.dtype != torch.float32:
                raise RuntimeError(f"`qk_mask` must be of type torch.float32, received {qk_mask.dtype}")
            if list(qk_mask.size()) != expected_qk_mask_shape:
                raise RuntimeError(f"Invalid shape for `qk_mask` (Expected {expected_qk_mask_shape}, got {list(qk_mask.size())}")

        expected_k_mask_shape = [b, src_seq_len, 1, 1]
        if k_mask is not None:
            if k_mask.dtype != torch.float32:
                raise RuntimeError(f"`k_mask` must be of type torch.float32, received {k_mask.dtype}")
            if list(k_mask.size()) != expected_k_mask_shape:
                raise RuntimeError(f"Invalid shape for `k_mask` (Expected {expected_k_mask_shape}, got {list(k_mask.size())}")

        attn, attn_weights = self._attention_fn(q, k, v, qk_mask, k_mask, return_weights)

        attn = attn.contiguous().view(b, self.d_v, ht, wt)
        attn = self.out_proj(attn)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def forward(self, q, k, v, **kwargs):
        return self._forward_impl(q, k, v, **kwargs)


class SelfAttention(MultiHeadAttention):
    # Vendored Apple API — q/k/v collapse to a single qkv input.
    def forward(self, qkv, **kwargs):  # ty: ignore[invalid-method-override]
        return super()._forward_impl(qkv, qkv, qkv, **kwargs)


def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """Unsqueeze 2D Linear weights to 4D Conv2d (1x1) weights for ANE-friendly modules.

    Vendored from ane_transformers/huggingface/distilbert.py. We rename the matching
    rule for our XLM-RoBERTa adaptation (see src/ane_xlm_roberta.py); the helper
    here just provides the unsqueeze logic. Use weight_transfer.py's mapping for
    the actual HF→ANE name translation.
    """
    for k in list(state_dict.keys()):
        v = state_dict[k]
        if k.endswith(".weight") and v.ndim == 2:
            state_dict[k] = v[:, :, None, None]
