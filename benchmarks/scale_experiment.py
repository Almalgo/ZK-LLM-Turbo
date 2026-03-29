"""Run the T1.6 CKKS scale reduction experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import tenseal as ts
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmarks.common import write_benchmark_report
from client.encryption.ckks_context import create_ckks_context, load_ckks_config


def _write_temp_config(scale: int) -> Path:
    base_cfg = yaml.safe_load((ROOT / "client/config/client_config.yaml").read_text())
    base_cfg["ckks"]["global_scale"] = scale
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(base_cfg, tmp)
    return Path(tmp.name)


def _measure_matmul_error(scale: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    context = create_ckks_context(global_scale_override=scale)
    x = rng.normal(0.0, 0.01, size=2048).astype(np.float32)
    W = rng.normal(0.0, 0.01, size=(2048, 2048)).astype(np.float32)

    expected = x @ W
    enc_x = ts.ckks_vector(context, x.tolist())
    enc_result = enc_x.mm(W.tolist())
    actual = np.array(enc_result.decrypt()[:2048], dtype=np.float32)
    error = actual - expected

    return {
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def _run_subprocess(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the T1.6 CKKS scale sweep experiment.")
    parser.add_argument("--samples", type=int, default=3, help="Samples for the HE matmul benchmark.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmups for the HE matmul benchmark.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    parser.add_argument(
        "--scales",
        type=int,
        nargs="*",
        default=[2**40, 2**35, 2**30],
        help="Absolute CKKS scale values to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/scale_experiment.json"),
        help="Output report path.",
    )
    args = parser.parse_args()

    results = []
    for scale in args.scales:
        config_path = _write_temp_config(scale)
        env = os.environ.copy()
        env["ZKLLM_CKKS_SCALE"] = str(scale)

        bench_cmd = [
            sys.executable,
            "benchmarks/bench_he_matmul.py",
            "--config-path",
            str(config_path),
            "--samples",
            str(args.samples),
            "--warmups",
            str(args.warmups),
            "--seed",
            str(args.seed),
            "--output",
            f"benchmarks/results/scale_{scale}_bench_he_matmul.json",
        ]
        test_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "client/tests/test_e2e_accuracy.py",
            "-q",
        ]

        bench_run = _run_subprocess(bench_cmd, env)
        test_run = _run_subprocess(test_cmd, env)
        error_metrics = _measure_matmul_error(scale, args.seed)

        bench_result_path = ROOT / f"benchmarks/results/scale_{scale}_bench_he_matmul.json"
        benchmark_summary = {}
        if bench_run.returncode == 0 and bench_result_path.exists():
            benchmark_summary = json.loads(bench_result_path.read_text())

        results.append(
            {
                "scale": scale,
                "scale_power": int(round(np.log2(scale))),
                "benchmark_exit_code": bench_run.returncode,
                "benchmark_stdout": bench_run.stdout,
                "benchmark_stderr": bench_run.stderr,
                "benchmark_results": benchmark_summary.get("results", []),
                "accuracy_exit_code": test_run.returncode,
                "accuracy_stdout": test_run.stdout,
                "accuracy_stderr": test_run.stderr,
                "matmul_error": error_metrics,
            }
        )
        config_path.unlink(missing_ok=True)

    output_path = write_benchmark_report(
        args.output,
        results=[],
        metadata={
            "experiment": "T1.6_scale_factor_reduction",
            "scales": args.scales,
            "samples": args.samples,
            "seed": args.seed,
            "runs": results,
        },
        ckks_params={**load_ckks_config(), "global_scale": "varies"},
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
