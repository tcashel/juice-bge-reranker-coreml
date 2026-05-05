"""
End-to-end PyTorch -> Core ML conversion for `BAAI/bge-reranker-base`.

Produces TWO `.mlpackage` artifacts in one run (Tier A shipping policy). Both
variants are converted from the same ANE-friendly port (Conv2d projections, BC1S
layout, LayerNormANE) so they share architecture, weights, and (B, 1, 1, S) input
shape — only the `compute_units` deployment hint differs:
  - `<output_dir>/bge-reranker-base-ane.mlpackage` — `compute_units=CPU_AND_NE`.
    Headline build; the 12-layer encoder backbone runs on the Apple Neural Engine.
  - `<output_dir>/bge-reranker-base-cpugpu.mlpackage` — `compute_units=CPU_AND_GPU`.
    Known-good fallback for when the ANE is unavailable (driver failures, Intel Macs).

Also writes:
  - `<output_dir>/tokenizer/`            — tokenizer files for the Swift consumer.
  - `<output_dir>/<variant>_provenance.json` — conversion provenance per variant.

Usage:
    pixi run python convert.py --output-dir ./build
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.ane_xlm_roberta import (
    ANEXLMRConfig,
    ANEXLMRobertaForSequenceClassification,
)
from src.provenance import Provenance
from src.weight_transfer import (
    assert_numerically_equivalent,
    transfer_weights,
)

DEFAULT_SOURCE_MODEL = "BAAI/bge-reranker-base"
DEFAULT_BATCH = 20
DEFAULT_SEQ_LENGTHS = (128, 256, 512)
DEFAULT_MAX_SEQ_LEN = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    p.add_argument("--output-dir", type=Path, default=Path("build"))
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument(
        "--hf-revision",
        default=None,
        help="Optional HF revision pin. Defaults to current main; the resolved SHA is recorded in provenance.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["ane", "cpugpu"],
        default=["ane", "cpugpu"],
        help="Which artifacts to produce. Default: both.",
    )
    p.add_argument(
        "--skip-equivalence-check",
        action="store_true",
        help="Skip the post-load HF<->ANE numerical equivalence assertion. Use only for debugging the port.",
    )
    return p.parse_args()


def resolve_revision(repo_id: str, revision: str | None) -> str:
    """Pin the revision to a concrete commit SHA so the artifact is reproducible."""
    api = HfApi()
    info = api.model_info(repo_id, revision=revision)
    if info.sha is None:
        raise RuntimeError(f"HfApi did not return a commit SHA for {repo_id}@{revision!r}")
    return info.sha


def save_tokenizer(tokenizer, dst: Path, source_model: str, source_revision: str) -> None:
    """Save tokenizer files mirroring the upstream layout.

    `tokenizer.save_pretrained` on a transformers 5.x fast tokenizer writes only
    `tokenizer.json` + `tokenizer_config.json` (the SPM model is bundled inside
    tokenizer.json). The Swift consumer's fast path is happy with that, but the
    upstream BAAI repo also ships `sentencepiece.bpe.model` and
    `special_tokens_map.json` as separate files for slow-tokenizer / belt-and-
    braces compatibility. Pull those from the source repo at the pinned revision
    so the published artifact matches the upstream file set.
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    tokenizer.save_pretrained(dst)
    if not (dst / "tokenizer.json").exists():
        raise RuntimeError(f"Tokenizer at {dst} is missing tokenizer.json — swift-transformers needs the fast format. Re-fetch with `use_fast=True`.")
    for upstream_file in ("sentencepiece.bpe.model", "special_tokens_map.json"):
        try:
            cached = hf_hub_download(repo_id=source_model, filename=upstream_file, revision=source_revision)
        except EntryNotFoundError:
            print(f"      [warning] {upstream_file} not present in {source_model}@{source_revision[:12]}; skipping")
            continue
        shutil.copy2(cached, dst / upstream_file)


def build_ane_model(hf_model, hf_config) -> ANEXLMRobertaForSequenceClassification:
    """Instantiate the ANE port and transfer HF weights into it."""
    ane_config = ANEXLMRConfig.from_hf(hf_config)
    ane_model = ANEXLMRobertaForSequenceClassification(ane_config)
    transfer_weights(hf_model, ane_model)
    return ane_model


def make_dummy_inputs(batch: int, seq: int, *, four_d: bool):
    """Build (input_ids, attention_mask) dummy tensors for tracing."""
    rng = torch.Generator().manual_seed(0)
    # vocab range is huge; pick something safely inside it. 250 002 is XLM-R vocab.
    ids = torch.randint(0, 250000, (batch, seq), generator=rng, dtype=torch.int32)
    mask = torch.ones((batch, seq), dtype=torch.int32)
    # Right-pad half the rows to exercise the mask path.
    half = batch // 2
    if half > 0 and seq > 4:
        ids[:half, seq // 2 :] = 1  # XLM-R pad_token_id
        mask[:half, seq // 2 :] = 0
    if four_d:
        ids = ids.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, S)
        mask = mask.unsqueeze(1).unsqueeze(1)
    return ids, mask


def trace_ane_model(
    ane_model: ANEXLMRobertaForSequenceClassification,
    *,
    batch: int,
    max_seq_len: int,
) -> torch.jit.ScriptModule:
    """torch.jit.trace the ANE port at (B, 1, 1, max_seq_len). Reused across variants."""
    ane_model.eval()
    input_ids, attn = make_dummy_inputs(batch, max_seq_len, four_d=True)
    with torch.no_grad():
        return torch.jit.trace(ane_model, (input_ids, attn))


def convert_variant(
    traced: torch.jit.ScriptModule,
    *,
    batch: int,
    seq_lengths: tuple[int, ...],
    max_seq_len: int,
    output_path: Path,
    compute_units: ct.ComputeUnit,
) -> None:
    """Convert the traced ANE port to a .mlpackage with the requested compute units.

    Both variants share architecture, weights, and (B, 1, 1, S) input shape; only
    the deployment hint differs. The cpuAndGPU build is the Tier-A safety net (used
    when ANE driver/loader fails); verify_ane.py only gates the ANE variant.
    """
    enumerated = ct.EnumeratedShapes(
        shapes=[(batch, 1, 1, s) for s in seq_lengths],
        default=(batch, 1, 1, max_seq_len),
    )
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=enumerated, dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=enumerated, dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logit", dtype=np.float32)],
        compute_units=compute_units,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save(str(output_path))


def main() -> int:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Resolving {args.source_model} revision...")
    sha = resolve_revision(args.source_model, args.hf_revision)
    print(f"      -> {sha}")

    print(f"[2/5] Loading HF model + tokenizer at {sha[:12]}...")
    hf_config = AutoConfig.from_pretrained(args.source_model, revision=sha)
    # `attn_implementation="eager"` avoids transformers 5.x's SDPA mask helper, which
    # can't be torch.jit.traced cleanly (IndexError in sdpa_mask on a 0-dim q_length).
    # Eager attention is mathematically identical to SDPA for inference.
    hf_model = AutoModelForSequenceClassification.from_pretrained(args.source_model, revision=sha, dtype=torch.float32, attn_implementation="eager")
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.source_model, revision=sha, use_fast=True)
    save_tokenizer(tokenizer, output_dir / "tokenizer", source_model=args.source_model, source_revision=sha)
    print(f"      tokenizer saved -> {output_dir / 'tokenizer'}")

    seq_lengths = tuple(s for s in DEFAULT_SEQ_LENGTHS if s <= args.max_seq_len)

    common_config = {
        "source_model": args.source_model,
        "source_revision": sha,
        "max_seq_len": args.max_seq_len,
        "seq_lengths": list(seq_lengths),
        "batch_size": DEFAULT_BATCH,
    }

    print("[3/5] Building ANE port + transferring weights...")
    ane_model = build_ane_model(hf_model, hf_config)
    if not args.skip_equivalence_check:
        print("       checking HF <-> ANE numerical equivalence...")
        ids, mask = make_dummy_inputs(2, 64, four_d=False)
        assert_numerically_equivalent(hf_model, ane_model, ids.to(torch.long), mask.to(torch.long))
        print("       OK")

    print("[4/5] Tracing ANE port...")
    traced = trace_ane_model(ane_model, batch=DEFAULT_BATCH, max_seq_len=args.max_seq_len)

    variant_compute_units = {
        "ane": ct.ComputeUnit.CPU_AND_NE,
        "cpugpu": ct.ComputeUnit.CPU_AND_GPU,
    }
    for variant in args.variants:
        artifact_path = output_dir / f"bge-reranker-base-{variant}.mlpackage"
        if artifact_path.exists():
            shutil.rmtree(artifact_path)
        print(f"       converting -> {artifact_path} (compute_units={variant_compute_units[variant].name})")
        convert_variant(
            traced,
            batch=DEFAULT_BATCH,
            seq_lengths=seq_lengths,
            max_seq_len=args.max_seq_len,
            output_path=artifact_path,
            compute_units=variant_compute_units[variant],
        )
        Provenance.build(
            source_repo=args.source_model,
            source_revision=sha,
            variant=variant,
            artifact_filename=artifact_path.name,
            config=common_config,
        ).write(output_dir / f"{variant}_provenance.json")

    print("[5/5] Done.")
    print(f"      Output dir: {output_dir.resolve()}")
    print("      Next: pixi run verify-ane build/bge-reranker-base-ane.mlpackage")
    print("            pixi run bench    build/bge-reranker-base-ane.mlpackage build/bge-reranker-base-cpugpu.mlpackage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
