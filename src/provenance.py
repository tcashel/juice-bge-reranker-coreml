"""
Conversion provenance: a JSON sidecar uploaded with each .mlpackage so the Juice
maintainer (and anyone auditing later) can reconstruct exactly which inputs and
toolchain produced a given artifact.

Schema lives at the bottom of this file.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata as _md
from pathlib import Path
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _machine_identifier() -> dict[str, str]:
    out: dict[str, str] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    # macOS-specific chip + product name. Best-effort; missing on non-mac.
    for key, sysctl_key in [
        ("chip", "machdep.cpu.brand_string"),
        ("model", "hw.model"),
    ]:
        try:
            value = subprocess.check_output(["sysctl", "-n", sysctl_key], text=True, stderr=subprocess.DEVNULL).strip()
            if value:
                out[key] = value
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return out


@dataclass
class ANEResidencyReport:
    """Filled in by verify_ane.py and written into provenance for the -ane variant."""

    verdict: str  # "pass" | "fail" | "skipped"
    total_ops: int
    ane_ops: int
    cpu_ops: int
    gpu_ops: int
    violations: list[dict[str, str]] = field(default_factory=list)


@dataclass
class BenchEntry:
    batch: int
    seq: int
    p50_ms: float
    p95_ms: float
    per_pair_p95_ms: float


@dataclass
class BenchReport:
    """Filled in by bench.py."""

    warmup_iters: int
    timed_iters: int
    entries: list[BenchEntry] = field(default_factory=list)


@dataclass
class Provenance:
    schema_version: str
    converted_at_utc: str
    source_repo: str
    source_revision: str
    variant: str  # "ane" | "cpugpu"
    artifact_filename: str
    package_versions: dict[str, str | None]
    machine: dict[str, str]
    config: dict[str, Any]
    ane_residency: ANEResidencyReport | None = None
    bench: BenchReport | None = None

    @classmethod
    def build(
        cls,
        *,
        source_repo: str,
        source_revision: str,
        variant: str,
        artifact_filename: str,
        config: dict[str, Any],
    ) -> Provenance:
        return cls(
            schema_version="1",
            converted_at_utc=_dt.datetime.now(_dt.UTC).isoformat(),
            source_repo=source_repo,
            source_revision=source_revision,
            variant=variant,
            artifact_filename=artifact_filename,
            package_versions={
                "torch": _pkg_version("torch"),
                "transformers": _pkg_version("transformers"),
                "coremltools": _pkg_version("coremltools"),
                "huggingface_hub": _pkg_version("huggingface_hub"),
                "sentencepiece": _pkg_version("sentencepiece"),
                "numpy": _pkg_version("numpy"),
            },
            machine=_machine_identifier(),
            config=config,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json())
