"""
ANE-friendly port of `XLMRobertaForSequenceClassification` for `BAAI/bge-reranker-base`.

Layout: (B, C, 1, S) channels-first throughout. All `nn.Linear` projections become
`nn.Conv2d(..., kernel_size=1)`; all `nn.LayerNorm` becomes `LayerNormANE`. Built on
the vendored Apple primitives in `vendor/ane_transformers/`.

XLM-RoBERTa specifics handled here (not BERT/DistilBERT):
- Vocab 250 002, position-embedding table size 514 with `padding_idx=1`.
- Position IDs are computed as `arange(S) + 2` masked by the attention mask
  (right-aligned real tokens, right-padded with `<pad>`=1). This is bit-equivalent
  to HF's `create_position_ids_from_input_ids` for the standard tokenizer output and
  avoids `cumsum`, which doesn't lower cleanly to ANE.
- Single-segment model: `token_type_embeddings` is size 1 and always indexed at 0.
- Classification head reads the `<s>` (position 0) hidden state, then
  `dense -> tanh -> out_proj` to a single logit. No pooler.

Numerical equivalence with the HF model is verified by `tests/test_numerical_equivalence.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vendor.ane_transformers.layer_norm import LayerNormANELoadable
from vendor.ane_transformers.multihead_attention import SelfAttention

# Tighter than HF's default 1e-12 — matches the FP16 ANE precision floor that
# Apple's distilbert reference uses.
ANE_LAYER_NORM_EPS = 1e-7


@dataclass
class ANEXLMRConfig:
    """Subset of HF XLMRobertaConfig that this port consumes.

    Populated from `transformers.AutoConfig.from_pretrained(...)`. Defaults match
    `BAAI/bge-reranker-base`'s config.json verbatim.
    """

    vocab_size: int = 250002
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    pad_token_id: int = 1
    layer_norm_eps: float = ANE_LAYER_NORM_EPS
    num_labels: int = 1

    @classmethod
    def from_hf(cls, hf_config) -> ANEXLMRConfig:
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            type_vocab_size=hf_config.type_vocab_size,
            pad_token_id=hf_config.pad_token_id,
            num_labels=getattr(hf_config, "num_labels", 1),
            # Override HF's 1e-12 with the FP16-safe value Apple uses.
            layer_norm_eps=ANE_LAYER_NORM_EPS,
        )


class ANEEmbeddings(nn.Module):
    """Sum of word/position/token_type embeddings, LayerNormANE, in BC1S layout."""

    def __init__(self, config: ANEXLMRConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.hidden_size = config.hidden_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = LayerNormANELoadable(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids:      (B, 1, 1, S) int32
        # attention_mask: (B, 1, 1, S) int32 (1 = real, 0 = pad)
        B = input_ids.shape[0]
        S = input_ids.shape[-1]

        ids = input_ids.view(B, S).long()
        mask = attention_mask.view(B, S).long()

        # Right-padded position IDs: real tokens get arange+2 (matches HF's
        # create_position_ids_from_input_ids), pad tokens get padding_idx.
        arange = torch.arange(S, device=input_ids.device, dtype=torch.long) + (self.padding_idx + 1)
        position_ids = arange.unsqueeze(0).expand(B, S) * mask + self.padding_idx * (1 - mask)

        # Token type is always zero for single-segment XLM-R.
        token_type_ids = torch.zeros((B, S), dtype=torch.long, device=input_ids.device)

        word_emb = self.word_embeddings(ids)  # (B, S, C)
        pos_emb = self.position_embeddings(position_ids)  # (B, S, C)
        ttype_emb = self.token_type_embeddings(token_type_ids)  # (B, S, C)
        emb = word_emb + pos_emb + ttype_emb  # (B, S, C)

        # BSC -> BC1S
        emb = emb.transpose(1, 2).unsqueeze(2)  # (B, C, 1, S)
        emb = self.LayerNorm(emb)  # (B, C, 1, S)
        return emb


class ANEEncoderLayer(nn.Module):
    """Post-LN RoBERTa encoder block in BC1S layout."""

    def __init__(self, config: ANEXLMRConfig):
        super().__init__()
        self.attention_self = SelfAttention(
            embed_dim=config.hidden_size,
            n_head=config.num_attention_heads,
            dropout=0.0,
        )
        self.attention_LayerNorm = LayerNormANELoadable(config.hidden_size, eps=config.layer_norm_eps)

        self.intermediate_dense = nn.Conv2d(config.hidden_size, config.intermediate_size, kernel_size=1)
        self.output_dense = nn.Conv2d(config.intermediate_size, config.hidden_size, kernel_size=1)
        self.output_LayerNorm = LayerNormANELoadable(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor, k_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, C, 1, S)
        # k_mask:        (B, S, 1, 1) additive (-1e4 for pad)
        attn_out, _ = self.attention_self(hidden_states, k_mask=k_mask, return_weights=False)
        attn_out = self.attention_LayerNorm(attn_out + hidden_states)

        ffn = self.intermediate_dense(attn_out)
        ffn = self.activation(ffn)
        ffn = self.output_dense(ffn)
        ffn = self.output_LayerNorm(ffn + attn_out)
        return ffn


class ANEClassificationHead(nn.Module):
    """RoBERTa-style classification head reading the <s> token in BC1S layout."""

    def __init__(self, config: ANEXLMRConfig):
        super().__init__()
        self.dense = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=1)
        self.out_proj = nn.Conv2d(config.hidden_size, config.num_labels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, C, 1, S) — take position 0 (the <s> token).
        cls_hidden = hidden_states[:, :, :, 0:1]  # (B, C, 1, 1)
        x = self.dense(cls_hidden)  # (B, C, 1, 1)
        x = torch.tanh(x)
        x = self.out_proj(x)  # (B, num_labels, 1, 1)
        return x.squeeze(-1).squeeze(-1)  # (B, num_labels)


class ANEXLMRobertaForSequenceClassification(nn.Module):
    """End-to-end ANE-friendly cross-encoder for `BAAI/bge-reranker-base`.

    Forward inputs:
        input_ids:      (B, 1, 1, S) int32, in {0..vocab_size-1}.
        attention_mask: (B, 1, 1, S) int32, in {0, 1}. 1 marks real tokens, 0 pad.

    Forward output:
        logit: (B, num_labels) float32. For bge-reranker-base, num_labels=1; the
        Swift consumer applies sigmoid to map to a relevance score in [0, 1].
    """

    def __init__(self, config: ANEXLMRConfig):
        super().__init__()
        self.config = config
        self.embeddings = ANEEmbeddings(config)
        self.encoder = nn.ModuleList([ANEEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.classifier = ANEClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Build additive key-mask in BC1S-attention shape (B, S, 1, 1).
        B = attention_mask.shape[0]
        S = attention_mask.shape[-1]
        mask_flat = attention_mask.view(B, S).float()
        k_mask = ((1.0 - mask_flat) * -1e4).view(B, S, 1, 1)

        hidden = self.embeddings(input_ids, attention_mask)
        for layer in self.encoder:
            hidden = layer(hidden, k_mask=k_mask)
        logit = self.classifier(hidden)
        return logit
