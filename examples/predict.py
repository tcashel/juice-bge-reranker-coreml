"""
Quick relevance-score check for the bge-reranker-base Core ML artifact.

Loads either a local `build/` artifact (pre-publish) or a tagged release from
`tcashel/bge-reranker-base-coreml` on Hugging Face Hub (post-publish), runs a
single (query, doc) pair through it, and prints the sigmoid-mapped relevance
score. Use this to smoke-test a freshly converted or freshly pulled artifact.

Requires macOS — `coremltools.models.MLModel` only loads on Apple platforms.

Usage:
    pixi run python examples/predict.py                                # uses build/bge-reranker-base-ane.mlpackage if present
    pixi run python examples/predict.py --source hub --tag v0.1-ane    # downloads from HF
    pixi run python examples/predict.py --source hub --tag v0.1-cpugpu # downloads cpugpu fallback
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from coremltools.models import MLModel
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

REPO_ID = "tcashel/bge-reranker-base-coreml"
PAD_TOKEN_ID = 1  # XLM-R <pad>
FIXED_BATCH = 20

# A short pair to smoke-test with; matches one of the fixtures in tests/test_numerical_equivalence.py.
DEFAULT_QUERY = "what is the capital of france?"
DEFAULT_DOC = "Paris is the capital of France."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("local", "hub"), default="local", help="Where to load the artifact from.")
    p.add_argument("--variant", choices=("ane", "cpugpu"), default="ane")
    p.add_argument("--tag", default="v0.1-ane", help="HF tag when --source=hub. Defaults to v0.1-ane.")
    p.add_argument("--build-dir", type=Path, default=Path("build"), help="Local build dir when --source=local.")
    p.add_argument("--seq", type=int, default=128, choices=(128, 256, 512))
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--doc", default=DEFAULT_DOC)
    return p.parse_args()


def resolve_artifact(args: argparse.Namespace) -> tuple[Path, Path]:
    """Returns (mlpackage_path, tokenizer_dir)."""
    if args.source == "local":
        mlpkg = args.build_dir / f"bge-reranker-base-{args.variant}.mlpackage"
        tok_dir = args.build_dir / "tokenizer"
        if not mlpkg.exists():
            raise SystemExit(f"{mlpkg} does not exist. Run `pixi run convert` first, or pass --source hub.")
        return mlpkg, tok_dir
    folder = Path(snapshot_download(repo_id=REPO_ID, revision=args.tag))
    return folder / "model.mlpackage", folder


def main() -> int:
    args = parse_args()
    mlpkg, tok_dir = resolve_artifact(args)
    print(f"loading tokenizer from {tok_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    assert tokenizer is not None, "AutoTokenizer.from_pretrained returned None"

    # Python's transformers exposes the paired-input call directly — it constructs
    # the XLM-R `<s> query </s></s> doc </s>` template internally. Swift consumers
    # have to build that template manually (see examples/swift/).
    enc = tokenizer(
        args.query,
        args.doc,
        padding="max_length",
        truncation=True,
        max_length=args.seq,
        return_tensors="np",
    )
    real_ids = enc["input_ids"][0].astype(np.int32)
    real_mask = enc["attention_mask"][0].astype(np.int32)

    # Pad up to the fixed batch=20 with all-pad rows; the model emits 20 logits but we only read row 0.
    ids = np.full((FIXED_BATCH, 1, 1, args.seq), PAD_TOKEN_ID, dtype=np.int32)
    mask = np.zeros((FIXED_BATCH, 1, 1, args.seq), dtype=np.int32)
    ids[0, 0, 0, :] = real_ids
    mask[0, 0, 0, :] = real_mask

    print(f"loading {mlpkg}")
    model = MLModel(str(mlpkg))
    out = model.predict({"input_ids": ids, "attention_mask": mask})
    logit = float(out["logit"][0, 0])
    score = 1.0 / (1.0 + math.exp(-logit))

    print(f"\nquery: {args.query}")
    print(f"doc:   {args.doc}")
    print(f"logit: {logit:+.4f}")
    print(f"score: {score:.4f}  (sigmoid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
