"""
Quality regression eval gate. Pulls MTEB Reranking datasets directly from HF,
scores every (query, candidate) pair through both the HF FP32 baseline and our
Core ML FP16 .mlpackage (ANE variant), computes nDCG@10 + MAP per task, and
stamps results into MODEL_CARD.md plus per-task YAML at .eval_results/<task>.yaml.

Pass criterion (gates publishing): |Δ nDCG@10| < 0.005 per task vs the FP32
reference. Exits non-zero on any failure.

Variant equivalence shortcut: the -ane and -cpugpu .mlpackages have bit-identical
FP16 weights (same conversion, only `compute_units` differs at load), so we score
the -ane variant only and note variant equivalence in the table.

Usage:
  pixi run -e eval eval                          # SciDocsRR only
  pixi run -e eval eval-full                     # all four MTEB Reranking tasks
  pixi run -e eval python eval/quality_regression.py --tasks scidocs-reranking --limit 50
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
DEFAULT_MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
EVAL_RESULTS_DIR = REPO_ROOT / ".eval_results"

SOURCE_MODEL = "BAAI/bge-reranker-base"
PUBLISHED_REPO_ID = "tcashel/bge-reranker-base-coreml"
FIXED_BATCH = 20  # matches convert.py / bench.py
PAD_TOKEN_ID = 1  # XLM-R <pad>
DEFAULT_MAX_SEQ = 512
DEFAULT_K = 10
HF_BATCH = 32  # forward-pass batch for the HF baseline (independent of FIXED_BATCH)
PASS_THRESHOLD = 0.005  # |Δ nDCG@10| gate

EVAL_SENTINEL_START = "<!-- EVAL:reranking -->"
EVAL_SENTINEL_END = "<!-- /EVAL:reranking -->"
MI_SENTINEL_START = "# MODEL_INDEX:auto-stamped — do not edit by hand"
MI_SENTINEL_END = "# /MODEL_INDEX"

# MTEB Reranking suite. `revision` should be a commit SHA after first run; until
# then we resolve and log it. `name` is the human-readable label used in YAML.
DATASETS: dict[str, dict[str, Any]] = {
    "scidocs-reranking": {
        "repo_id": "mteb/scidocs-reranking",
        "name": "SciDocs Reranking",
        "revision": None,
    },
    "askubuntudupquestions-reranking": {
        "repo_id": "mteb/askubuntudupquestions-reranking",
        "name": "AskUbuntu Duplicate Questions Reranking",
        "revision": None,
    },
    "mind-small-reranking": {
        "repo_id": "mteb/mind_small_reranking",
        "name": "MIND Small Reranking",
        "revision": None,
    },
    "stackoverflowdupquestions-reranking": {
        "repo_id": "mteb/stackoverflowdupquestions-reranking",
        "name": "StackOverflow Duplicate Questions Reranking",
        "revision": None,
    },
}
DEFAULT_TASKS = ["scidocs-reranking"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help=f"Task names from {sorted(DATASETS)} or 'all-reranking'. Default: {DEFAULT_TASKS}.",
    )
    p.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit each dataset to first N queries (for fast iteration).",
    )
    p.add_argument(
        "--update-model-card",
        type=Path,
        default=DEFAULT_MODEL_CARD,
        help="MODEL_CARD.md to stamp the eval table into (between EVAL:reranking sentinels).",
    )
    p.add_argument(
        "--no-update-model-card",
        dest="update_model_card",
        action="store_const",
        const=None,
        help="Skip MODEL_CARD update.",
    )
    p.add_argument(
        "--no-hf-baseline",
        action="store_true",
        help="Skip the HF FP32 baseline. Without a baseline the gate cannot fire; useful for prototyping.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "mps"),
        help="Torch device for the HF baseline. CPU for determinism; MPS for speed.",
    )
    return p.parse_args()


# -- Metrics (pure numpy) -----------------------------------------------------


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = DEFAULT_K) -> float:
    order = np.argsort(-scores)
    ranked = labels[order][:k]
    discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    dcg = float((ranked * discounts).sum())
    ideal = np.sort(labels)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float((ideal * ideal_discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    ranked = labels[order]
    if int(ranked.sum()) == 0:
        return 0.0
    hits = 0
    precisions: list[float] = []
    for i, lab in enumerate(ranked.tolist()):
        if lab == 1:
            hits += 1
            precisions.append(hits / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0


# -- Data ---------------------------------------------------------------------


def normalize_row(row: dict) -> tuple[str, list[str], list[str]]:
    """Extract (query, positive, negative) handling minor schema differences across MTEB datasets."""
    query = row.get("query") or row.get("question") or row.get("title")
    pos = row.get("positive") or row.get("positives") or row.get("relevant_documents") or []
    neg = row.get("negative") or row.get("negatives") or row.get("non_relevant_documents") or []
    if query is None:
        raise SystemExit(f"row has no recognizable query field: keys={list(row)}")
    if not isinstance(pos, list) or not isinstance(neg, list):
        raise SystemExit(f"expected list-of-strings for positive/negative; got {type(pos).__name__} / {type(neg).__name__}")
    return str(query), [str(s) for s in pos], [str(s) for s in neg]


# -- Scoring ------------------------------------------------------------------


def score_hf_batched(
    model: Any, tokenizer: Any, queries: list[str], docs: list[str], max_seq: int, device: str, timing: dict[str, float] | None = None
) -> np.ndarray:
    n = len(queries)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, HF_BATCH):
        end = min(start + HF_BATCH, n)
        enc = tokenizer(
            queries[start:end],
            docs[start:end],
            padding="max_length",
            truncation="only_second",
            max_length=max_seq,
            return_tensors="pt",
        )
        enc_on_device = {k: v.to(device) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(**enc_on_device).logits.squeeze(-1).cpu().numpy()
        if timing is not None:
            timing["hf_inference_s"] += time.perf_counter() - t0
        out[start:end] = logits
    return out


def score_coreml_batched(
    coreml_model: ct.models.MLModel, tokenizer: Any, queries: list[str], docs: list[str], max_seq: int, timing: dict[str, float] | None = None
) -> np.ndarray:
    """Score (query, doc) pairs through the Core ML .mlpackage (fixed batch=20, BC1S layout)."""
    n = len(queries)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, FIXED_BATCH):
        end = min(start + FIXED_BATCH, n)
        actual = end - start
        enc = tokenizer(
            queries[start:end],
            docs[start:end],
            padding="max_length",
            truncation="only_second",
            max_length=max_seq,
            return_tensors="np",
        )
        ids_full = np.full((FIXED_BATCH, max_seq), PAD_TOKEN_ID, dtype=np.int32)
        mask_full = np.zeros((FIXED_BATCH, max_seq), dtype=np.int32)
        ids_full[:actual] = enc["input_ids"].astype(np.int32)
        mask_full[:actual] = enc["attention_mask"].astype(np.int32)
        # ANE port input shape contract: (B, 1, 1, S).
        ids_4d = ids_full[:, None, None, :]
        mask_4d = mask_full[:, None, None, :]
        t0 = time.perf_counter()
        result = coreml_model.predict({"input_ids": ids_4d, "attention_mask": mask_4d})
        if timing is not None:
            timing["coreml_inference_s"] += time.perf_counter() - t0
            timing["coreml_predict_calls"] += 1
        out[start:end] = result["logit"][:actual, 0]
    return out


# -- Per-task driver ----------------------------------------------------------


def evaluate_task(
    task: str,
    *,
    coreml_model: ct.models.MLModel,
    hf_model: Any,
    tokenizer: Any,
    max_seq: int,
    limit: int | None,
    device: str,
) -> dict:
    spec = DATASETS[task]
    repo_id = spec["repo_id"]
    print(f"\n[task={task}] loading {repo_id} (split=test, revision={spec['revision']})...", flush=True)
    ds = load_dataset(repo_id, split="test", revision=spec["revision"])
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    n_queries = len(ds)
    print(f"  {n_queries} queries", flush=True)

    coreml_ndcg: list[float] = []
    coreml_map: list[float] = []
    hf_ndcg: list[float] = []
    hf_map: list[float] = []
    timing: dict[str, float] = {"coreml_inference_s": 0.0, "coreml_predict_calls": 0.0, "hf_inference_s": 0.0, "total_pairs": 0.0}
    wall_t0 = time.perf_counter()

    for q_idx, row in enumerate(ds):
        query, pos, neg = normalize_row(row)
        if not pos or not neg:
            continue
        pool = pos + neg
        labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        queries_rep = [query] * len(pool)
        timing["total_pairs"] += len(pool)

        cml_scores = score_coreml_batched(coreml_model, tokenizer, queries_rep, pool, max_seq, timing=timing)
        coreml_ndcg.append(ndcg_at_k(cml_scores, labels))
        coreml_map.append(average_precision(cml_scores, labels))

        if hf_model is not None:
            hf_scores = score_hf_batched(hf_model, tokenizer, queries_rep, pool, max_seq, device, timing=timing)
            hf_ndcg.append(ndcg_at_k(hf_scores, labels))
            hf_map.append(average_precision(hf_scores, labels))

        if (q_idx + 1) % 50 == 0:
            cml_mean = float(np.mean(coreml_ndcg))
            elapsed = time.perf_counter() - wall_t0
            cml_per_pair_ms = 1000.0 * timing["coreml_inference_s"] / max(timing["total_pairs"], 1)
            msg = f"  [{q_idx + 1}/{n_queries}] coreml nDCG@10={cml_mean:.4f}  cml/pair={cml_per_pair_ms:5.1f}ms  elapsed={elapsed:5.0f}s"
            if hf_ndcg:
                msg += f"  hf nDCG@10={float(np.mean(hf_ndcg)):.4f}  Δ={cml_mean - float(np.mean(hf_ndcg)):+.4f}"
            print(msg, flush=True)

    wall_s = time.perf_counter() - wall_t0
    pairs = max(timing["total_pairs"], 1)
    out: dict[str, Any] = {
        "task": task,
        "repo_id": repo_id,
        "name": spec["name"],
        "revision": spec["revision"],
        "n_queries": len(coreml_ndcg),
        "n_pairs": int(timing["total_pairs"]),
        "coreml": {
            "ndcg_at_10": float(np.mean(coreml_ndcg)) if coreml_ndcg else 0.0,
            "map": float(np.mean(coreml_map)) if coreml_map else 0.0,
        },
        "timing": {
            "wall_s": wall_s,
            "coreml_inference_s": timing["coreml_inference_s"],
            "coreml_predict_calls": int(timing["coreml_predict_calls"]),
            "coreml_per_pair_ms": 1000.0 * timing["coreml_inference_s"] / pairs,
            "coreml_per_call_ms": 1000.0 * timing["coreml_inference_s"] / max(timing["coreml_predict_calls"], 1),
            "hf_inference_s": timing["hf_inference_s"],
            "hf_per_pair_ms": 1000.0 * timing["hf_inference_s"] / pairs,
        },
    }
    if hf_ndcg:
        out["hf_fp32"] = {
            "ndcg_at_10": float(np.mean(hf_ndcg)),
            "map": float(np.mean(hf_map)),
        }
        out["delta_ndcg_at_10"] = out["coreml"]["ndcg_at_10"] - out["hf_fp32"]["ndcg_at_10"]
    return out


# -- Output: .eval_results/<task>.yaml ---------------------------------------


def write_eval_yaml(result: dict) -> Path:
    """Write per-task result to .eval_results/<task>.yaml in HF model-index style."""
    EVAL_RESULTS_DIR.mkdir(exist_ok=True)
    path = EVAL_RESULTS_DIR / f"{result['task']}.yaml"
    today = datetime.now(UTC).date().isoformat()
    cml = result["coreml"]
    revision = result.get("revision") or "<not pinned>"

    lines = [
        "# Generated by eval/quality_regression.py — do not edit by hand.",
        "# Per HF model-index spec: https://huggingface.co/docs/hub/model-cards",
        "# Variant equivalence: scores apply to both -ane and -cpugpu (FP16 weights are bit-identical).",
        "model-index:",
        "  - name: bge-reranker-base-coreml",
        "    results:",
        "      - task:",
        "          type: text-ranking",
        "          name: Reranking",
        "        dataset:",
        f"          type: {result['repo_id']}",
        f"          name: {result['name']}",
        "          split: test",
        f"          revision: {revision}",
        "        metrics:",
        "          - type: ndcg_at_10",
        "            name: nDCG@10",
        f"            value: {cml['ndcg_at_10']:.4f}",
        "          - type: map",
        "            name: MAP",
        f"            value: {cml['map']:.4f}",
        "        source:",
        f"          name: bge-reranker-base-coreml quality regression eval ({today})",
        "          url: https://github.com/tcashel/juice-bge-reranker-coreml",
    ]
    if "hf_fp32" in result:
        hf = result["hf_fp32"]
        lines.extend(
            [
                "# Reference (BAAI/bge-reranker-base FP32 on the same split, attn_implementation=eager):",
                f"#   fp32_ndcg_at_10:   {hf['ndcg_at_10']:.4f}",
                f"#   fp32_map:          {hf['map']:.4f}",
                f"#   delta_ndcg_at_10:  {result['delta_ndcg_at_10']:+.4f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    return path


# -- Output: MODEL_CARD.md table + frontmatter model-index -------------------


def render_eval_table(results: list[dict]) -> str:
    """Markdown comparison table for the EVAL:reranking sentinel block."""
    lines = [
        "### MTEB Reranking — FP32 reference vs Core ML FP16",
        "",
        "_Variant equivalence: FP16 weights are bit-identical between `-ane` and `-cpugpu`; both inherit these numbers._",
        "",
        "| Task | n queries | FP32 nDCG@10 | Core ML nDCG@10 | Δ nDCG@10 | FP32 MAP | Core ML MAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        cml = r["coreml"]
        hf = r.get("hf_fp32")
        if hf is not None:
            lines.append(
                f"| {r['task']} | {r['n_queries']} | {hf['ndcg_at_10']:.4f} | "
                f"{cml['ndcg_at_10']:.4f} | {r['delta_ndcg_at_10']:+.4f} | "
                f"{hf['map']:.4f} | {cml['map']:.4f} |"
            )
        else:
            lines.append(f"| {r['task']} | {r['n_queries']} | _no baseline_ | {cml['ndcg_at_10']:.4f} | n/a | _no baseline_ | {cml['map']:.4f} |")
    lines.extend(
        [
            "",
            f"**Pass criterion:** `|Δ nDCG@10| < {PASS_THRESHOLD}` per task. "
            'FP32 baseline is `BAAI/bge-reranker-base` loaded with `attn_implementation="eager"`.',
        ]
    )
    return "\n".join(lines) + "\n"


def render_model_index_block(results: list[dict]) -> str:
    """YAML model-index block for MODEL_CARD frontmatter (between MI sentinels)."""
    lines = [
        MI_SENTINEL_START,
        "model-index:",
        "  - name: bge-reranker-base-coreml",
        "    results:",
    ]
    for r in results:
        cml = r["coreml"]
        revision = r.get("revision") or "<not pinned>"
        lines.extend(
            [
                "      - task:",
                "          type: text-ranking",
                "          name: Reranking",
                "        dataset:",
                f"          type: {r['repo_id']}",
                f"          name: {r['name']}",
                "          split: test",
                f"          revision: {revision}",
                "        metrics:",
                "          - type: ndcg_at_10",
                "            name: nDCG@10",
                f"            value: {cml['ndcg_at_10']:.4f}",
                "          - type: map",
                "            name: MAP",
                f"            value: {cml['map']:.4f}",
            ]
        )
    lines.append(MI_SENTINEL_END)
    return "\n".join(lines) + "\n"


def update_model_card(path: Path, table_md: str, model_index_yaml: str) -> None:
    """Stamp the EVAL:reranking body table + the MODEL_INDEX frontmatter block.

    Mirrors `bench.py:update_model_card`'s sentinel-partition pattern. Both placeholder
    blocks (body and frontmatter) must already exist in the file; this function only
    rewrites their contents.
    """
    text = path.read_text()
    # Body table
    if EVAL_SENTINEL_START not in text or EVAL_SENTINEL_END not in text:
        raise SystemExit(
            f"MODEL_CARD.md missing {EVAL_SENTINEL_START} / {EVAL_SENTINEL_END} sentinels; "
            "add a placeholder under '## Quality regression eval' before running with --update-model-card."
        )
    before, _, rest = text.partition(EVAL_SENTINEL_START)
    _, _, after = rest.partition(EVAL_SENTINEL_END)
    text = f"{before}{EVAL_SENTINEL_START}\n{table_md}{EVAL_SENTINEL_END}{after}"
    # Frontmatter model-index
    if MI_SENTINEL_START not in text or MI_SENTINEL_END not in text:
        raise SystemExit(
            f"MODEL_CARD.md missing {MI_SENTINEL_START} / {MI_SENTINEL_END} sentinels in frontmatter; "
            "add an empty model-index placeholder before running with --update-model-card."
        )
    before, _, rest = text.partition(MI_SENTINEL_START)
    _, _, after = rest.partition(MI_SENTINEL_END)
    text = f"{before}{model_index_yaml}{after}"
    path.write_text(text)


# -- Main ---------------------------------------------------------------------


def resolve_tasks(requested: list[str]) -> list[str]:
    if requested == ["all-reranking"]:
        return list(DATASETS)
    unknown = [t for t in requested if t not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}. Available: {sorted(DATASETS)}.")
    return requested


def main() -> int:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)

    artifact = args.build_dir / "bge-reranker-base-ane.mlpackage"
    if not artifact.exists():
        raise SystemExit(f"missing {artifact}. Run `pixi run convert` first.")
    tokenizer_dir = args.build_dir / "tokenizer"
    if not tokenizer_dir.exists():
        raise SystemExit(f"missing {tokenizer_dir}. Run `pixi run convert` first.")

    print(f"loading Core ML artifact: {artifact}")
    coreml_model = ct.models.MLModel(str(artifact), compute_units=ct.ComputeUnit.CPU_AND_NE)

    print(f"loading tokenizer from {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    assert tokenizer is not None, "AutoTokenizer.from_pretrained returned None"

    hf_model: Any | None = None
    if not args.no_hf_baseline:
        print(f"loading HF FP32 baseline: {SOURCE_MODEL} on {args.device} (attn_implementation='eager')")
        loaded = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL, dtype=torch.float32, attn_implementation="eager")
        assert loaded is not None, "AutoModelForSequenceClassification.from_pretrained returned None"
        hf_model = loaded.eval().to(args.device)

    results: list[dict] = []
    for task in tasks:
        result = evaluate_task(
            task,
            coreml_model=coreml_model,
            hf_model=hf_model,
            tokenizer=tokenizer,
            max_seq=args.max_seq,
            limit=args.limit,
            device=args.device,
        )
        results.append(result)
        yaml_path = write_eval_yaml(result)
        print(f"  wrote {yaml_path.relative_to(REPO_ROOT)}")

    # Summary
    print("\n=== Summary ===")
    for r in results:
        cml = r["coreml"]
        hf = r.get("hf_fp32")
        t = r["timing"]
        print(f"\n{r['task']}:  n_queries={r['n_queries']}  n_pairs={r['n_pairs']}  wall={t['wall_s']:.1f}s")
        print(f"  Core ML (ANE):  nDCG@10={cml['ndcg_at_10']:.4f}  MAP={cml['map']:.4f}")
        print(
            f"                  inference={t['coreml_inference_s']:.1f}s  per-pair={t['coreml_per_pair_ms']:.2f}ms  "
            f"per-predict-call={t['coreml_per_call_ms']:.2f}ms"
        )
        if hf is not None:
            print(f"  HF FP32 ({args.device}):  nDCG@10={hf['ndcg_at_10']:.4f}  MAP={hf['map']:.4f}  Δ nDCG@10={r['delta_ndcg_at_10']:+.4f}")
            print(f"                  inference={t['hf_inference_s']:.1f}s  per-pair={t['hf_per_pair_ms']:.2f}ms")

    if args.update_model_card is not None:
        table_md = render_eval_table(results)
        mi_yaml = render_model_index_block(results)
        update_model_card(args.update_model_card, table_md, mi_yaml)
        print(f"\nupdated {args.update_model_card.relative_to(REPO_ROOT)}")

    # Gate
    failures: list[str] = []
    for r in results:
        if "delta_ndcg_at_10" not in r:
            continue  # skipped baseline; cannot gate
        if abs(r["delta_ndcg_at_10"]) >= PASS_THRESHOLD:
            failures.append(f"{r['task']}: |Δ nDCG@10|={abs(r['delta_ndcg_at_10']):.4f} >= {PASS_THRESHOLD}")
    if failures:
        print("\nQUALITY REGRESSION GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if hf_model is None:
        print("\n[warning] --no-hf-baseline was set; quality gate did not run.")
        return 0
    print(f"\nQuality gate OK: |Δ nDCG@10| < {PASS_THRESHOLD} on every task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
