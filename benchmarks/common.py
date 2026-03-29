"""Shared helpers for benchmark scripts and fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, UTC
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

RESULTS_DIR = ROOT / "benchmarks" / "results"


def load_ckks_params(config_path: Path | None = None) -> dict:
    """Load CKKS parameters from the client config."""
    config_path = config_path or ROOT / "client" / "config" / "client_config.yaml"
    return yaml.safe_load(config_path.read_text())["ckks"]


def current_git_sha() -> str:
    """Return the current git SHA, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def benchmark_operation(fn, samples: int, warmups: int = 1) -> list[float]:
    """Run an operation repeatedly and return sample timings in milliseconds."""
    for _ in range(max(warmups, 0)):
        fn()

    measurements = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        measurements.append((time.perf_counter() - start) * 1000)
    return measurements


def summarize_samples(
    name: str,
    samples_ms: list[float],
    metadata: dict | None = None,
) -> dict:
    """Build one benchmark result record."""
    result = {
        "name": name,
        "mean_ms": round(mean(samples_ms), 3),
        "std_ms": round(stdev(samples_ms), 3) if len(samples_ms) > 1 else 0.0,
        "samples": len(samples_ms),
        "raw_samples_ms": [round(sample, 3) for sample in samples_ms],
    }
    if metadata:
        result["metadata"] = metadata
    return result


def write_benchmark_report(
    output_path: Path,
    results: list[dict],
    metadata: dict | None = None,
) -> Path:
    """Write benchmark results in the shared JSON envelope."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": current_git_sha(),
        "ckks_params": load_ckks_params(),
        "results": results,
    }
    if metadata:
        report["metadata"] = metadata

    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return output_path


def seeded_rng(seed: int) -> np.random.Generator:
    """Return a deterministic RNG for reproducible benchmarks."""
    return np.random.default_rng(seed)


def require_server(base_url: str) -> None:
    """Fail fast with a helpful message if the inference server is unavailable."""
    try:
        with urllib.request.urlopen(base_url, timeout=5):
            return
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"Server not reachable at {base_url}. Start it with: python -m server.server"
        ) from exc
