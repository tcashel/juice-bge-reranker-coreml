"""
Verify that every op in a converted .mlpackage is dispatched to the Apple Neural Engine.

Uses `coremltools.models.compute_plan.MLComputePlan` (added in coremltools 8.1, requires
macOS 14.4+). Compiles the .mlpackage to .mlmodelc, walks the program function, and
collects any op whose `preferred_compute_device` is not `NeuralEngine`. Exits non-zero
on any violation so this can run as a CI gate.

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


def _device_name(usage) -> str:
    if usage is None:
        return "Unknown"
    device = usage.preferred_compute_device
    return device if isinstance(device, str) else getattr(device, "name", str(device))


def verify(mlpackage_path: Path) -> ANEResidencyReport:
    """Compile the .mlpackage and assert every op is ANE-resident."""
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
    violations: list[dict[str, str]] = []

    for op in main_fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        device = _device_name(usage)
        counts[device] += 1
        if device != "NeuralEngine":
            violations.append(
                {
                    "op_type": getattr(op, "operator_type", getattr(op, "type", "unknown")),
                    "op_name": getattr(op, "name", ""),
                    "device": device,
                }
            )

    total = sum(counts.values())
    report = ANEResidencyReport(
        verdict="pass" if not violations else "fail",
        total_ops=total,
        ane_ops=counts.get("NeuralEngine", 0),
        cpu_ops=counts.get("CPU", 0),
        gpu_ops=counts.get("GPU", 0),
        violations=violations,
    )
    return report


def render_human_summary(report: ANEResidencyReport, mlpackage_path: Path) -> str:
    lines = [
        f"ANE residency report: {mlpackage_path}",
        f"  total ops: {report.total_ops}",
        f"  ANE: {report.ane_ops}    CPU: {report.cpu_ops}    GPU: {report.gpu_ops}",
        f"  verdict: {report.verdict.upper()}",
    ]
    if report.violations:
        lines.append("  violations:")
        for v in report.violations:
            lines.append(f"    - [{v['device']}] {v['op_type']} ({v['op_name']})")
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

    report = verify(args.mlpackage)
    print(render_human_summary(report, args.mlpackage))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))

    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
