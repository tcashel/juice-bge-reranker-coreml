"""
HF `XLMRobertaForSequenceClassification` -> `ANEXLMRobertaForSequenceClassification`
state_dict translator.

Two transformations apply:

1. Name remapping. HF's nested module path
   `roberta.encoder.layer.{i}.attention.self.query.weight` becomes the flat ANE name
   `encoder.{i}.attention_self.q_proj.weight`, and similarly for every other tensor.

2. Linear -> Conv2d reshape. Every 2D `nn.Linear.weight` of shape `(out, in)` is reshaped
   to `(out, in, 1, 1)` to match `nn.Conv2d(kernel_size=1).weight`. Biases pass through
   unchanged.

LayerNorm bias/scale order inversion (HF: `x*w + b`, ANE: `(x + b)*w`) is handled by
the `correct_for_bias_scale_order_inversion` pre-hook registered inside
`LayerNormANELoadable`, so we do NOT do that adjustment here. Loading the renamed
state_dict triggers the hook automatically per LayerNorm instance.
"""

from __future__ import annotations

import torch

from src.ane_xlm_roberta import ANEXLMRobertaForSequenceClassification


def build_state_dict_remap(num_hidden_layers: int) -> dict[str, str]:
    """Return a HF-key -> ANE-key mapping for every parameter in the model."""
    m: dict[str, str] = {
        "roberta.embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
        "roberta.embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "roberta.embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
        "roberta.embeddings.LayerNorm.weight": "embeddings.LayerNorm.weight",
        "roberta.embeddings.LayerNorm.bias": "embeddings.LayerNorm.bias",
        "classifier.dense.weight": "classifier.dense.weight",
        "classifier.dense.bias": "classifier.dense.bias",
        "classifier.out_proj.weight": "classifier.out_proj.weight",
        "classifier.out_proj.bias": "classifier.out_proj.bias",
    }

    for i in range(num_hidden_layers):
        hf_prefix = f"roberta.encoder.layer.{i}"
        ane_prefix = f"encoder.{i}"
        per_layer = {
            f"{hf_prefix}.attention.self.query.weight": f"{ane_prefix}.attention_self.q_proj.weight",
            f"{hf_prefix}.attention.self.query.bias": f"{ane_prefix}.attention_self.q_proj.bias",
            f"{hf_prefix}.attention.self.key.weight": f"{ane_prefix}.attention_self.k_proj.weight",
            f"{hf_prefix}.attention.self.key.bias": f"{ane_prefix}.attention_self.k_proj.bias",
            f"{hf_prefix}.attention.self.value.weight": f"{ane_prefix}.attention_self.v_proj.weight",
            f"{hf_prefix}.attention.self.value.bias": f"{ane_prefix}.attention_self.v_proj.bias",
            f"{hf_prefix}.attention.output.dense.weight": f"{ane_prefix}.attention_self.out_proj.weight",
            f"{hf_prefix}.attention.output.dense.bias": f"{ane_prefix}.attention_self.out_proj.bias",
            f"{hf_prefix}.attention.output.LayerNorm.weight": f"{ane_prefix}.attention_LayerNorm.weight",
            f"{hf_prefix}.attention.output.LayerNorm.bias": f"{ane_prefix}.attention_LayerNorm.bias",
            f"{hf_prefix}.intermediate.dense.weight": f"{ane_prefix}.intermediate_dense.weight",
            f"{hf_prefix}.intermediate.dense.bias": f"{ane_prefix}.intermediate_dense.bias",
            f"{hf_prefix}.output.dense.weight": f"{ane_prefix}.output_dense.weight",
            f"{hf_prefix}.output.dense.bias": f"{ane_prefix}.output_dense.bias",
            f"{hf_prefix}.output.LayerNorm.weight": f"{ane_prefix}.output_LayerNorm.weight",
            f"{hf_prefix}.output.LayerNorm.bias": f"{ane_prefix}.output_LayerNorm.bias",
        }
        m.update(per_layer)
    return m


# HF -> ANE: keys whose 2D Linear.weight needs (out, in) -> (out, in, 1, 1).
# All other params (embedding tables, LayerNorm weight/bias, biases) keep their shape.
_LINEAR_WEIGHT_SUFFIXES = (
    "attention_self.q_proj.weight",
    "attention_self.k_proj.weight",
    "attention_self.v_proj.weight",
    "attention_self.out_proj.weight",
    "intermediate_dense.weight",
    "output_dense.weight",
    "classifier.dense.weight",
    "classifier.out_proj.weight",
)


def _is_linear_weight(ane_key: str) -> bool:
    return any(ane_key.endswith(suffix) for suffix in _LINEAR_WEIGHT_SUFFIXES)


def remap_state_dict(
    hf_state: dict[str, torch.Tensor],
    num_hidden_layers: int,
) -> dict[str, torch.Tensor]:
    """Build an ANE-shaped state_dict from an HF XLMRobertaForSequenceClassification one.

    Unknown HF keys (e.g. `roberta.pooler.*` if present, dropout buffers, etc.) are
    silently skipped — the pooler is not used by the classification head.
    """
    remap = build_state_dict_remap(num_hidden_layers)
    out: dict[str, torch.Tensor] = {}
    for hf_key, hf_tensor in hf_state.items():
        ane_key = remap.get(hf_key)
        if ane_key is None:
            continue
        tensor = hf_tensor
        if _is_linear_weight(ane_key) and tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).unsqueeze(-1).contiguous()
        out[ane_key] = tensor
    return out


def transfer_weights(
    hf_model: torch.nn.Module,
    ane_model: ANEXLMRobertaForSequenceClassification,
) -> None:
    """In-place: load HF weights into the ANE model.

    Raises if any ANE parameter is missing from the remapped state_dict (strict load).
    Unexpected keys are tolerated because the pooler may exist in some HF checkpoints
    and is unused for sequence classification.
    """
    hf_state = hf_model.state_dict()
    ane_state = remap_state_dict(hf_state, ane_model.config.num_hidden_layers)
    missing, unexpected = ane_model.load_state_dict(ane_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when transferring HF -> ANE weights: {sorted(missing)}")
    # `unexpected` should be empty after our explicit remap; if not, surface it loudly
    # since it points at a remap-table mistake.
    if unexpected:
        raise RuntimeError(f"Unexpected keys reached load_state_dict: {sorted(unexpected)}")


def assert_numerically_equivalent(
    hf_model: torch.nn.Module,
    ane_model: ANEXLMRobertaForSequenceClassification,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Run both models in eval mode and compare logits.

    Args:
        hf_model:       HF XLMRobertaForSequenceClassification (B, S) inputs.
        ane_model:      Our ANE port (B, 1, 1, S) inputs.
        input_ids:      (B, S) int64 tensor (HF convention).
        attention_mask: (B, S) int64 tensor.
        atol/rtol:      Tolerances for `torch.testing.assert_close` on the FP32 logits.
    """
    hf_model.eval()
    ane_model.eval()
    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, attention_mask=attention_mask)
        # transformers >= 5 returns a SequenceClassifierOutput dataclass; older returns a tuple.
        hf_logits = getattr(hf_out, "logits", None)
        if hf_logits is None:
            hf_logits = hf_out[0]

        ane_input_ids = input_ids.to(torch.int32).unsqueeze(1).unsqueeze(1)
        ane_attn = attention_mask.to(torch.int32).unsqueeze(1).unsqueeze(1)
        ane_logits = ane_model(ane_input_ids, ane_attn)

    torch.testing.assert_close(ane_logits, hf_logits, atol=atol, rtol=rtol)
