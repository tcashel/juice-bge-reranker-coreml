"""
Numerical equivalence between HF and our ANE port.

Two layers of test:

1. **Pure-Python (no model download):** verifies the HF→ANE state_dict remap covers
   every named parameter HF emits and produces the right Conv2d shapes.

2. **End-to-end (downloads BAAI/bge-reranker-base on first run, cached after):**
   loads the HF model, instantiates the ANE port, transfers weights, runs both on
   16 fixed query/doc pairs, and asserts logits agree within 1e-3.

The end-to-end test is opt-in by default — skip without `JUICE_RUN_DOWNLOAD_TESTS=1`
in env, since downloading the 1.1 GB checkpoint is too slow for casual `pixi run test`
runs. CI / pre-publish should set the variable.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.ane_xlm_roberta import ANEXLMRConfig, ANEXLMRobertaForSequenceClassification
from src.weight_transfer import (
    assert_numerically_equivalent,
    build_state_dict_remap,
    remap_state_dict,
    transfer_weights,
)

SOURCE_MODEL = "BAAI/bge-reranker-base"

FIXED_PAIRS = [
    ("what is the capital of france?", "Paris is the capital of France."),
    ("how does a transformer work?", "A transformer uses self-attention to process sequences in parallel."),
    ("python list comprehension", "List comprehensions provide a concise way to create lists in Python."),
    ("error: cannot find module", "ModuleNotFoundError occurs when an imported module is not installed."),
    ("recipe for chocolate cake", "The plot of Hamlet centers on a Danish prince."),
    ("apple silicon vs intel", "Apple Silicon offers better performance per watt than Intel."),
    ("git rebase tutorial", "git rebase replays commits onto another base."),
    ("rust borrow checker", "The borrow checker prevents data races at compile time."),
    ("kubernetes pod restart", "Pods can be restarted by deleting them; the deployment will recreate."),
    ("sql window function", "Window functions perform calculations across rows related to the current row."),
    ("pytorch dataloader workers", "num_workers > 0 enables parallel data loading."),
    ("does the moon have water", "Yes, lunar polar regions contain water ice in shadowed craters."),
    ("typescript generic constraint", "Use `extends` to constrain a generic type parameter."),
    ("css flexbox align-items", "align-items aligns flex items along the cross axis."),
    ("vim exit insert mode", "Press Esc to leave insert mode and return to normal mode."),
    ("query unrelated to docs", "this string is intentionally unrelated to the query above"),
]


def test_remap_table_covers_all_layers():
    """The remap table must include every per-layer key for a 12-layer model."""
    remap = build_state_dict_remap(num_hidden_layers=12)
    layer_keys_per_layer = 16  # Q,K,V (w+b) + out (w+b) + 2 LN (w+b) + intermediate (w+b) + output (w+b) + LN (w+b)
    expected_total = 9 + 12 * layer_keys_per_layer  # 9 root keys (embeddings + classifier) + per-layer
    assert len(remap) == expected_total

    # Each ANE target key should be unique (no collisions).
    targets = list(remap.values())
    assert len(set(targets)) == len(targets), "duplicate ANE-key targets in remap"


def test_remap_reshapes_linear_weights_to_conv2d():
    """Synthetic state_dict round-trip: 2D weights become 4D, biases stay 1D."""
    fake_hf_state = {
        "roberta.embeddings.word_embeddings.weight": torch.zeros(250002, 768),
        "roberta.embeddings.position_embeddings.weight": torch.zeros(514, 768),
        "roberta.embeddings.token_type_embeddings.weight": torch.zeros(1, 768),
        "roberta.embeddings.LayerNorm.weight": torch.ones(768),
        "roberta.embeddings.LayerNorm.bias": torch.zeros(768),
        "roberta.encoder.layer.0.attention.self.query.weight": torch.zeros(768, 768),
        "roberta.encoder.layer.0.attention.self.query.bias": torch.zeros(768),
        "roberta.encoder.layer.0.intermediate.dense.weight": torch.zeros(3072, 768),
        "roberta.encoder.layer.0.output.dense.weight": torch.zeros(768, 3072),
        "classifier.dense.weight": torch.zeros(768, 768),
        "classifier.dense.bias": torch.zeros(768),
        "classifier.out_proj.weight": torch.zeros(1, 768),
        "classifier.out_proj.bias": torch.zeros(1),
    }
    out = remap_state_dict(fake_hf_state, num_hidden_layers=1)

    assert out["embeddings.word_embeddings.weight"].shape == (250002, 768)
    assert out["embeddings.LayerNorm.bias"].shape == (768,)
    assert out["encoder.0.attention_self.q_proj.weight"].shape == (768, 768, 1, 1)
    assert out["encoder.0.attention_self.q_proj.bias"].shape == (768,)
    assert out["encoder.0.intermediate_dense.weight"].shape == (3072, 768, 1, 1)
    assert out["encoder.0.output_dense.weight"].shape == (768, 3072, 1, 1)
    assert out["classifier.dense.weight"].shape == (768, 768, 1, 1)
    assert out["classifier.out_proj.weight"].shape == (1, 768, 1, 1)


def test_ane_model_forward_shape():
    """ANE port instantiates and runs at the expected output shape with random inputs."""
    config = ANEXLMRConfig(
        # Tiny to keep the test fast; the geometry just has to round-trip.
        vocab_size=200,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=64,
        type_vocab_size=1,
        pad_token_id=1,
        num_labels=1,
    )
    model = ANEXLMRobertaForSequenceClassification(config).eval()
    batch, seq = 4, 16
    ids = torch.randint(0, 200, (batch, 1, 1, seq), dtype=torch.int32)
    mask = torch.ones((batch, 1, 1, seq), dtype=torch.int32)
    with torch.no_grad():
        out = model(ids, mask)
    assert out.shape == (batch, config.num_labels)
    assert out.dtype == torch.float32


# ---------- end-to-end (downloads HF weights) ----------


def _download_tests_enabled() -> bool:
    return os.environ.get("JUICE_RUN_DOWNLOAD_TESTS", "0") == "1"


@pytest.mark.skipif(
    not _download_tests_enabled(),
    reason="Set JUICE_RUN_DOWNLOAD_TESTS=1 to enable end-to-end equivalence (downloads ~1 GB).",
)
def test_hf_to_ane_numerical_equivalence_on_fixed_pairs():
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(SOURCE_MODEL)
    hf_model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, use_fast=True)

    queries = [q for q, _ in FIXED_PAIRS]
    docs = [d for _, d in FIXED_PAIRS]
    assert tokenizer is not None, "AutoTokenizer.from_pretrained returned None"
    enc = tokenizer(
        queries,
        docs,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    ane_model = ANEXLMRobertaForSequenceClassification(ANEXLMRConfig.from_hf(hf_config))
    transfer_weights(hf_model, ane_model)

    assert_numerically_equivalent(
        hf_model,
        ane_model,
        enc["input_ids"],
        enc["attention_mask"],
        atol=1e-3,
        rtol=1e-3,
    )
