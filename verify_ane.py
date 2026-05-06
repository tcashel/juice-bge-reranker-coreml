"""
Verify the residency profile of a converted .mlpackage matches the documented baseline.

Uses `coremltools.models.compute_plan.MLComputePlan` (added in coremltools 8.1, requires
macOS 14.4+). Compiles the .mlpackage to .mlmodelc, walks the program function, and
inspects each op's `preferred_compute_device`. Constants (`const` ops) have no device
usage and are correctly excluded from the count.

**Policy.** A small, fixed set of boundary ops physically can't run on the ANE for
this architecture — most importantly the embedding gather over the 250k-entry XLM-R
vocabulary, plus position-id arithmetic, mask construction, and a few int<->float
casts at the encoder boundary. These are pinned in `EXPECTED_CPU_FINGERPRINT` below.
The script exits 0 if and only if:

  - No op dispatches to GPU (the ANE build should not fall through to GPU; that
    would mean ANE refused entirely).
  - The CPU op type+count breakdown matches `EXPECTED_CPU_FINGERPRINT` exactly.

Any drift — a new CPU op type, more occurrences of an expected type, or fewer than
expected — is flagged so the maintainer can investigate. This is strictly more
informative than a blanket "any non-NE op is FAIL" rule, which gives the same FAIL
verdict whether 31 ops drifted to CPU or 31,000 did.

Usage:
    pixi run python verify_ane.py build/bge-reranker-base-ane.mlpackage
    pixi run python verify_ane.py --json-out build/ane_residency.json build/bge-reranker-base-ane.mlpackage
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import coremltools as ct

from src.provenance import ANEResidencyReport

# Pareto-frontier CPU dispatches for the bge-reranker-base ANE port. These are
# physically forced off the ANE by op-coverage limits at the encoder boundary
# (the 250k-vocab gather is the headline case) — they're not porting choices.
# Any drift from this fingerprint is treated as a regression and fails the gate.
EXPECTED_CPU_FINGERPRINT: dict[str, int] = {
    "ios18.cast": 5,  # int<->float boundary casts around mask/position-id arithmetic
    "ios18.gather": 4,  # 250k-row embedding lookup (ANE refuses very-large vocabs)
    "ios18.reshape": 3,  # boundary tensor reshapes feeding the encoder
    "ios18.add": 3,  # position-id / mask construction
    "ios18.sub": 2,  # ditto
    "ios18.mul": 2,  # ditto
    "shape": 2,  # dynamic shape extraction at input
    "select": 2,
    "range_1d": 1,  # arange for position IDs
    "ios18.expand_dims": 1,
    "ios18.concat": 1,
    "ios18.equal": 1,
    "ios18.real_div": 1,
    "tile": 1,
    "fill": 1,
    "ios18.greater_equal": 1,
}


def _device_name(device) -> str:
    """Map a coremltools MLComputeDevice subclass to a short name."""
    if device is None:
        return "Unknown"
    cls = type(device).__name__
    if "NeuralEngine" in cls:
        return "NeuralEngine"
    if "CPU" in cls:
        return "CPU"
    if "GPU" in cls:
        return "GPU"
    return cls


def _fingerprint_violations(actual: Counter[str], expected: dict[str, int]) -> list[dict[str, str]]:
    """Return a list of structured drift entries between the actual and expected CPU fingerprints.

    Each entry is `{"op_type": str, "actual": str, "expected": str}`. Empty list = exact match.
    """
    drift: list[dict[str, str]] = []
    op_types = set(actual) | set(expected)
    for op_type in sorted(op_types):
        a = actual.get(op_type, 0)
        e = expected.get(op_type, 0)
        if a != e:
            drift.append({"op_type": op_type, "actual": str(a), "expected": str(e)})
    return drift


def verify(mlpackage_path: Path) -> tuple[ANEResidencyReport, Counter[str], list[dict[str, str]]]:
    """Compile the .mlpackage and check residency against the documented fingerprint.

    Returns (report, actual CPU op breakdown, fingerprint drift list). The drift list
    is empty when the actual breakdown exactly matches `EXPECTED_CPU_FINGERPRINT`.
    """
    if not mlpackage_path.exists():
        raise FileNotFoundError(mlpackage_path)

    # Loading the MLModel triggers compilation; get_compiled_model_path() returns
    # the .mlmodelc directory we then hand to MLComputePlan.
    model = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled_path = model.get_compiled_model_path()

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled_path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        raise RuntimeError(
            f"{mlpackage_path}: model structure has no `program` — only mlprogram-format Core ML packages support per-op compute-plan inspection."
        )
    main_fn = program.functions["main"]

    counts: Counter[str] = Counter()
    cpu_op_types: Counter[str] = Counter()
    gpu_violations: list[dict[str, str]] = []

    for op in main_fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            # `const` ops and other compile-time values have no dispatch.
            counts["Const"] += 1
            continue
        device = _device_name(usage.preferred_compute_device)
        counts[device] += 1
        op_type = op.operator_name
        if device == "CPU":
            cpu_op_types[op_type] += 1
        elif device == "GPU":
            # GPU dispatch on the -ane build means the ANE refused entirely;
            # always fail, regardless of the CPU fingerprint.
            gpu_violations.append({"op_type": op_type, "device": device})

    fingerprint_drift = _fingerprint_violations(cpu_op_types, EXPECTED_CPU_FINGERPRINT)
    drift_as_violations = [{"op_type": d["op_type"], "device": "CPU", "actual": d["actual"], "expected": d["expected"]} for d in fingerprint_drift]
    total = sum(counts.values())
    report = ANEResidencyReport(
        verdict="pass" if not (fingerprint_drift or gpu_violations) else "fail",
        total_ops=total,
        ane_ops=counts.get("NeuralEngine", 0),
        cpu_ops=counts.get("CPU", 0),
        gpu_ops=counts.get("GPU", 0),
        violations=gpu_violations + drift_as_violations,
    )
    return report, cpu_op_types, fingerprint_drift


def render_human_summary(report: ANEResidencyReport, cpu_op_types: Counter[str], drift: list[dict[str, str]], mlpackage_path: Path) -> str:
    dispatched = report.ane_ops + report.cpu_ops + report.gpu_ops
    consts = report.total_ops - dispatched
    lines = [
        f"ANE residency report: {mlpackage_path}",
        f"  total ops: {report.total_ops}  ({consts} const + {dispatched} dispatched)",
        f"  dispatched: NE={report.ane_ops}  CPU={report.cpu_ops}  GPU={report.gpu_ops}",
        f"  verdict: {report.verdict.upper()}",
    ]
    if cpu_op_types:
        lines.append("  CPU op type breakdown:")
        for op_type, n in cpu_op_types.most_common():
            expected = EXPECTED_CPU_FINGERPRINT.get(op_type, 0)
            marker = "ok" if n == expected else f"DRIFT (expected {expected})"
            lines.append(f"    {op_type:>26} x {n:<3}  {marker}")
    if drift:
        lines.append("  fingerprint drift (this is what failed the gate):")
        for d in drift:
            lines.append(f"    {d['op_type']}: actual={d['actual']}, expected={d['expected']}")
    if report.gpu_ops > 0:
        lines.append("  GPU dispatches detected — the -ane build should never fall through to GPU.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mlpackage", type=Path, help="Path to a .mlpackage to verify.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the report as JSON.",
    )
    args = parser.parse_args()

    report, cpu_op_types, drift = verify(args.mlpackage)
    print(render_human_summary(report, cpu_op_types, drift, args.mlpackage))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))

    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
