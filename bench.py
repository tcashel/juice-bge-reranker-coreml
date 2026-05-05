"""
Benchmark a converted .mlpackage across (batch, seq) grid and emit a Markdown
latency table.

For each (batch, seq) pair: 50 warmup iterations (Apple's first-call ANE
specialization is one-shot) + 100 timed iterations via time.perf_counter.

Because the converted model has fixed batch = 20, smaller batch sweeps pad up to
20 with `<pad>` tokens and report two figures:
  - p50 / p95: wall-clock per call (the latency the Swift caller observes).
  - per_pair_p95: p95 / actual_batch — the cost amortized across the candidates
    that were not pad. This is the headline number ADR 0006 cares about.

Pass criterion (ANE variant): p95(batch=20, seq=256) < 200 ms AND per_pair_p95 < 15 ms.

Usage:
    pixi run python bench.py build/bge-reranker-base-ane.mlpackage
    pixi run python bench.py --variants ane:build/...-ane.mlpackage cpugpu:build/...-cpugpu.mlpackage \\
        --update-model-card MODEL_CARD.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import coremltools as ct
import numpy as np

from src.provenance import BenchEntry, BenchReport

DEFAULT_BATCHES = (1, 4, 10, 20)
DEFAULT_SEQS = (128, 256, 512)
WARMUP_ITERS = 50
TIMED_ITERS = 100
FIXED_BATCH = 20
PAD_TOKEN_ID = 1  # XLM-R <pad>

# ANE-variant pass criteria (matches ADR 0006 budget).
ANE_HEADLINE_BATCH = 20
ANE_HEADLINE_SEQ = 256
ANE_P95_BUDGET_MS = 200.0
ANE_PER_PAIR_P95_BUDGET_MS = 15.0


def _compute_units_for(variant: str) -> ct.ComputeUnit:
    if variant == "ane":
        return ct.ComputeUnit.CPU_AND_NE
    if variant == "cpugpu":
        return ct.ComputeUnit.CPU_AND_GPU
    raise ValueError(f"unknown variant: {variant!r}")


def _make_inputs(actual_batch: int, seq: int) -> dict[str, np.ndarray]:
    """Build (input_ids, attention_mask) padded up to FIXED_BATCH x seq."""
    rng = np.random.default_rng(seed=0)
    ids = rng.integers(0, 250000, size=(FIXED_BATCH, seq), dtype=np.int32)
    mask = np.ones((FIXED_BATCH, seq), dtype=np.int32)
    if actual_batch < FIXED_BATCH:
        ids[actual_batch:, :] = PAD_TOKEN_ID
        mask[actual_batch:, :] = 0
    # The ANE variant takes (B, 1, 1, S); the cpuAndGPU variant takes (B, S).
    return {"_ids_2d": ids, "_mask_2d": mask}


def percentile(values: list[float], p: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def benchmark_one(model: ct.models.MLModel, batch: int, seq: int) -> BenchEntry:
    inputs = _make_inputs(batch, seq)
    spec_inputs = model.get_spec().description.input
    # Detect rank from the spec — ANE variant is rank-4, cpuAndGPU is rank-2.
    rank = len(spec_inputs[0].type.multiArrayType.shape) or len(spec_inputs[0].type.multiArrayType.enumeratedShapes.shapes[0].shape)
    if rank == 4:
        ids = inputs["_ids_2d"][:, None, None, :]
        mask = inputs["_mask_2d"][:, None, None, :]
    else:
        ids = inputs["_ids_2d"]
        mask = inputs["_mask_2d"]

    payload = {"input_ids": ids, "attention_mask": mask}

    for _ in range(WARMUP_ITERS):
        model.predict(payload)

    samples_ms: list[float] = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        model.predict(payload)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = statistics.median(samples_ms)
    p95 = percentile(samples_ms, 95)
    return BenchEntry(
        batch=batch,
        seq=seq,
        p50_ms=p50,
        p95_ms=p95,
        per_pair_p95_ms=p95 / max(batch, 1),
    )


def benchmark_variant(
    mlpackage_path: Path,
    variant: str,
    batches: Iterable[int],
    seqs: Iterable[int],
) -> BenchReport:
    print(f"\n[bench] variant={variant} -> {mlpackage_path}")
    model = ct.models.MLModel(
        str(mlpackage_path),
        compute_units=_compute_units_for(variant),
    )
    entries: list[BenchEntry] = []
    for seq in seqs:
        for batch in batches:
            entry = benchmark_one(model, batch, seq)
            print(
                f"  batch={batch:>2} seq={seq:>3}: "
                f"p50={entry.p50_ms:7.2f} ms  p95={entry.p95_ms:7.2f} ms  "
                f"per_pair_p95={entry.per_pair_p95_ms:6.2f} ms"
            )
            entries.append(entry)
    return BenchReport(warmup_iters=WARMUP_ITERS, timed_iters=TIMED_ITERS, entries=entries)


def render_markdown_table(report: BenchReport, title: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| batch | seq | p50 (ms) | p95 (ms) | per-pair p95 (ms) |",
        "|------:|----:|---------:|---------:|------------------:|",
    ]
    for e in report.entries:
        lines.append(f"| {e.batch} | {e.seq} | {e.p50_ms:.2f} | {e.p95_ms:.2f} | {e.per_pair_p95_ms:.2f} |")
    return "\n".join(lines) + "\n"


def update_model_card(model_card: Path, blocks: dict[str, str]) -> None:
    """Replace each `<!-- BENCH:<variant> -->` block in MODEL_CARD.md with rendered tables.

    The template must include matched `<!-- BENCH:<variant> --> ... <!-- /BENCH:<variant> -->`
    sentinels. We rewrite the content between them.
    """
    text = model_card.read_text()
    for variant, table in blocks.items():
        start = f"<!-- BENCH:{variant} -->"
        end = f"<!-- /BENCH:{variant} -->"
        if start not in text or end not in text:
            print(f"warning: MODEL_CARD.md missing {start} ... {end} sentinels; skipping {variant}")
            continue
        before, _, rest = text.partition(start)
        _, _, after = rest.partition(end)
        text = f"{before}{start}\n{table}{end}{after}"
    model_card.write_text(text)


def passes_ane_budget(report: BenchReport) -> bool:
    for e in report.entries:
        if e.batch == ANE_HEADLINE_BATCH and e.seq == ANE_HEADLINE_SEQ:
            return e.p95_ms < ANE_P95_BUDGET_MS and e.per_pair_p95_ms < ANE_PER_PAIR_P95_BUDGET_MS
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mlpackage",
        type=Path,
        nargs="?",
        help="Single .mlpackage to bench (variant inferred from filename: -ane / -cpugpu).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        metavar="VARIANT:PATH",
        help="Multiple variants to bench, e.g. ane:build/x-ane.mlpackage cpugpu:build/x-cpugpu.mlpackage.",
    )
    parser.add_argument(
        "--update-model-card",
        type=Path,
        default=None,
        help="MODEL_CARD.md to stamp tables into via <!-- BENCH:<variant> --> sentinels.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _infer_variant(path: Path) -> str:
    name = path.name
    if "-ane" in name:
        return "ane"
    if "-cpugpu" in name:
        return "cpugpu"
    raise SystemExit(f"Cannot infer variant from filename {name!r}; pass --variants explicitly.")


def main() -> int:
    args = parse_args()

    targets: list[tuple[str, Path]] = []
    if args.variants:
        for spec in args.variants:
            if ":" not in spec:
                raise SystemExit(f"--variants entry must be VARIANT:PATH, got {spec!r}")
            variant, path_s = spec.split(":", 1)
            targets.append((variant, Path(path_s)))
    elif args.mlpackage:
        targets.append((_infer_variant(args.mlpackage), args.mlpackage))
    else:
        raise SystemExit("Pass either a positional mlpackage or --variants entries.")

    reports: dict[str, BenchReport] = {}
    for variant, path in targets:
        reports[variant] = benchmark_variant(path, variant, DEFAULT_BATCHES, DEFAULT_SEQS)

    blocks = {variant: render_markdown_table(report, title=f"Variant: `{variant}`") for variant, report in reports.items()}
    print()
    for table in blocks.values():
        print(table)

    if args.update_model_card is not None:
        update_model_card(args.update_model_card, blocks)
        print(f"updated {args.update_model_card}")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps({v: asdict(r) for v, r in reports.items()}, indent=2, sort_keys=True))

    if "ane" in reports:
        if passes_ane_budget(reports["ane"]):
            print(
                f"ANE budget OK: p95(batch={ANE_HEADLINE_BATCH}, seq={ANE_HEADLINE_SEQ}) "
                f"< {ANE_P95_BUDGET_MS} ms and per_pair_p95 < {ANE_PER_PAIR_P95_BUDGET_MS} ms"
            )
            return 0
        print(f"ANE budget FAIL at (batch={ANE_HEADLINE_BATCH}, seq={ANE_HEADLINE_SEQ}). Iterate on the port before publishing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
