"""Probe whether current OpenFHE path appears to use GPU execution."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.he_backend import create_context, encrypt_vector, matmul


def _query_gpu() -> dict:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()}

    line = proc.stdout.strip().splitlines()[0]
    util_str, mem_str = [part.strip() for part in line.split(",", maxsplit=1)]
    return {
        "ok": True,
        "utilization_gpu": int(util_str),
        "memory_used_mib": int(mem_str),
    }


def _gpu_symbol_probe() -> dict:
    try:
        import openfhe  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"importable": False, "error": str(exc), "gpu_symbols": []}

    names = dir(openfhe)
    gpu_symbols = sorted([name for name in names if "gpu" in name.lower() or "cuda" in name.lower()])
    return {
        "importable": True,
        "gpu_symbols": gpu_symbols,
    }


def _run_openfhe_small_matmul() -> dict:
    old_backend = os.environ.get("ZKLLM_HE_BACKEND")
    os.environ["ZKLLM_HE_BACKEND"] = "openfhe"
    try:
        context = create_context(
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            global_scale=2**40,
            use_galois_keys=True,
            use_relin_keys=True,
        )
        x = np.random.normal(0.0, 0.01, size=256).astype(np.float32)
        w = np.random.normal(0.0, 0.01, size=(256, 128)).astype(np.float32)
        _ = matmul(encrypt_vector(context, x), w)
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        if old_backend is None:
            os.environ.pop("ZKLLM_HE_BACKEND", None)
        else:
            os.environ["ZKLLM_HE_BACKEND"] = old_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe GPU feasibility for current OpenFHE runtime")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/t3_gpu_feasibility.json"),
    )
    args = parser.parse_args()

    symbol_probe = _gpu_symbol_probe()
    gpu_before = _query_gpu()
    run_result = _run_openfhe_small_matmul()
    gpu_after = _query_gpu()

    if gpu_before.get("ok") and gpu_after.get("ok"):
        util_delta = gpu_after["utilization_gpu"] - gpu_before["utilization_gpu"]
        mem_delta = gpu_after["memory_used_mib"] - gpu_before["memory_used_mib"]
    else:
        util_delta = None
        mem_delta = None

    gpu_path_usable = bool(symbol_probe.get("gpu_symbols")) and bool(run_result.get("ok"))

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "symbol_probe": symbol_probe,
        "gpu_before": gpu_before,
        "openfhe_small_matmul": run_result,
        "gpu_after": gpu_after,
        "utilization_delta": util_delta,
        "memory_delta_mib": mem_delta,
        "gpu_path_usable": gpu_path_usable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"gpu_path_usable={gpu_path_usable}")


if __name__ == "__main__":
    main()
