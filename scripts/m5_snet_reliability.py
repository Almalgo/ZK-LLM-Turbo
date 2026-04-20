#!/usr/bin/env python3
"""Milestone 5 reliability and recovery harness for daemon passthrough."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
import sys
import time

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.common import write_benchmark_report
from client.encryption.ckks_context import create_ckks_context
from common.he_backend import encrypt_vector, serialize_public_context, serialize_vector


def _headers(auth_token: str | None) -> dict[str, str]:
    if not auth_token:
        return {"Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def _run_single_attempt(
    *,
    attempt_idx: int,
    base_url: str,
    session_endpoint: str,
    layer_endpoint: str,
    layer_idx: int,
    vector_size: int,
    timeout: float,
    auth_token: str | None,
    seed: int,
    global_start: float,
) -> dict:
    started = time.perf_counter()
    session_url = f"{base_url}{session_endpoint}"
    layer_url = f"{base_url}{layer_endpoint}"
    headers = _headers(auth_token)

    result = {
        "attempt": attempt_idx,
        "started_offset_ms": round((started - global_start) * 1000.0, 3),
        "ok": False,
    }

    try:
        context = create_ckks_context()
        public_context_b64 = base64.b64encode(serialize_public_context(context)).decode("utf-8")

        t0 = time.perf_counter()
        session_resp = requests.post(
            session_url,
            json={"public_context_b64": public_context_b64},
            headers=headers,
            timeout=timeout,
        )
        result["session_elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        result["session_status_code"] = session_resp.status_code
        session_resp.raise_for_status()
        session_id = session_resp.json().get("session_id")
        if not session_id:
            raise RuntimeError("Session response missing 'session_id'")

        rng = np.random.default_rng(seed + attempt_idx)
        plaintext_vec = rng.normal(0.0, 0.01, size=vector_size).astype(np.float32)
        encrypted_vec = encrypt_vector(context, plaintext_vec.tolist())
        encrypted_vec_b64 = base64.b64encode(serialize_vector(encrypted_vec)).decode("utf-8")

        layer_payload = {
            "session_id": session_id,
            "layer_idx": layer_idx,
            "operation": "qkv",
            "encrypted_vectors_b64": [encrypted_vec_b64],
        }

        t0 = time.perf_counter()
        layer_resp = requests.post(
            layer_url,
            json=layer_payload,
            headers=headers,
            timeout=timeout,
        )
        result["layer_elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        result["layer_status_code"] = layer_resp.status_code
        layer_resp.raise_for_status()
        layer_json = layer_resp.json()
        enc_results = layer_json.get("encrypted_results_b64")
        if not isinstance(enc_results, list):
            raise RuntimeError("Layer response missing list field 'encrypted_results_b64'")
        if len(enc_results) != 3:
            raise RuntimeError(f"Expected 3 encrypted results for qkv, got {len(enc_results)}")

        result["result_count"] = len(enc_results)
        result["server_elapsed_ms"] = layer_json.get("elapsed_ms")
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)

    result["total_elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    result["ended_offset_ms"] = round((time.perf_counter() - global_start) * 1000.0, 3)
    return result


def _recovery_summary(attempts: list[dict]) -> dict:
    ordered = sorted(attempts, key=lambda item: item["started_offset_ms"])

    failure_streaks = 0
    recovered_streaks = 0
    unrecovered_streaks = 0
    in_failure = False
    first_failure_offset = 0.0
    recovery_samples_ms: list[float] = []

    for item in ordered:
        if not item["ok"]:
            if not in_failure:
                in_failure = True
                failure_streaks += 1
                first_failure_offset = item["started_offset_ms"]
            continue

        if in_failure:
            recovered_streaks += 1
            in_failure = False
            recovery_samples_ms.append(item["ended_offset_ms"] - first_failure_offset)

    if in_failure:
        unrecovered_streaks = 1

    return {
        "failure_streaks": failure_streaks,
        "recovered_streaks": recovered_streaks,
        "unrecovered_streaks": unrecovered_streaks,
        "recovery_events": len(recovery_samples_ms),
        "recovery_p50_ms": _percentile(recovery_samples_ms, 50),
        "recovery_p95_ms": _percentile(recovery_samples_ms, 95),
        "recovery_samples_ms": [round(value, 3) for value in recovery_samples_ms],
    }


def _write_reliability_report(
    output: Path,
    *,
    metadata: dict,
    attempts: list[dict],
    min_success_rate: float,
) -> tuple[Path, bool]:
    success = [item for item in attempts if item["ok"]]
    failures = [item for item in attempts if not item["ok"]]
    success_rate = (len(success) / len(attempts)) if attempts else 0.0

    success_total_ms = [item["total_elapsed_ms"] for item in success]
    success_session_ms = [item.get("session_elapsed_ms") for item in success if item.get("session_elapsed_ms") is not None]
    success_layer_ms = [item.get("layer_elapsed_ms") for item in success if item.get("layer_elapsed_ms") is not None]

    reliability_metadata = dict(metadata)
    reliability_metadata.update(
        {
            "attempts": len(attempts),
            "successes": len(success),
            "failures": len(failures),
            "success_rate": round(success_rate, 6),
            "min_success_rate": min_success_rate,
            "total_p50_ms": _percentile(success_total_ms, 50),
            "total_p95_ms": _percentile(success_total_ms, 95),
            "session_p50_ms": _percentile(success_session_ms, 50),
            "session_p95_ms": _percentile(success_session_ms, 95),
            "layer_p50_ms": _percentile(success_layer_ms, 50),
            "layer_p95_ms": _percentile(success_layer_ms, 95),
            "error_examples": [item.get("error", "") for item in failures[:5]],
        }
    )
    passed = success_rate >= min_success_rate

    result = {
        "name": "m5_snet_reliability",
        "status": "pass" if passed else "fail",
        "metadata": reliability_metadata,
    }
    report_path = write_benchmark_report(output, [result], metadata=reliability_metadata)
    return report_path, passed


def _write_recovery_report(
    output: Path,
    *,
    metadata: dict,
    attempts: list[dict],
) -> tuple[Path, bool]:
    recovery = _recovery_summary(attempts)
    recovery_metadata = dict(metadata)
    recovery_metadata.update(recovery)

    passed = recovery["unrecovered_streaks"] == 0
    result = {
        "name": "m5_snet_recovery",
        "status": "pass" if passed else "fail",
        "metadata": recovery_metadata,
    }
    report_path = write_benchmark_report(output, [result], metadata=recovery_metadata)
    return report_path, passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 5 reliability/recovery harness")
    parser.add_argument("--base-url", required=True, help="Daemon base URL, e.g. http://host:7000")
    parser.add_argument("--session-endpoint", default="/api/session", help="Session endpoint path")
    parser.add_argument("--layer-endpoint", default="/api/layer", help="Layer endpoint path")
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index for qkv call")
    parser.add_argument("--vector-size", type=int, default=2048, help="Input vector size")
    parser.add_argument("--attempts", type=int, default=20, help="Total number of attempts")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds")
    parser.add_argument("--auth-token", default=None, help="Optional bearer token")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.95,
        help="Minimum reliability success rate for pass/fail",
    )
    parser.add_argument(
        "--reliability-output",
        type=Path,
        default=Path("benchmarks/results/m5_reliability_sepolia.json"),
        help="Reliability artifact output path",
    )
    parser.add_argument(
        "--recovery-output",
        type=Path,
        default=Path("benchmarks/results/m5_recovery_sepolia.json"),
        help="Recovery artifact output path",
    )
    args = parser.parse_args()

    if args.attempts <= 0:
        raise SystemExit("--attempts must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")

    base_url = args.base_url.rstrip("/")
    run_started = time.perf_counter()
    run_metadata = {
        "run_utc": datetime.now(UTC).isoformat(),
        "daemon_base_url": base_url,
        "session_endpoint": args.session_endpoint,
        "layer_endpoint": args.layer_endpoint,
        "layer_idx": args.layer_idx,
        "operation": "qkv",
        "vector_size": args.vector_size,
        "attempts": args.attempts,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "timeout_seconds": args.timeout,
    }

    attempt_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _run_single_attempt,
                attempt_idx=attempt,
                base_url=base_url,
                session_endpoint=args.session_endpoint,
                layer_endpoint=args.layer_endpoint,
                layer_idx=args.layer_idx,
                vector_size=args.vector_size,
                timeout=args.timeout,
                auth_token=args.auth_token,
                seed=args.seed,
                global_start=run_started,
            )
            for attempt in range(args.attempts)
        ]
        for fut in as_completed(futures):
            attempt_results.append(fut.result())

    run_metadata["run_total_elapsed_ms"] = round((time.perf_counter() - run_started) * 1000.0, 3)

    reliability_path, reliability_passed = _write_reliability_report(
        args.reliability_output,
        metadata=run_metadata,
        attempts=attempt_results,
        min_success_rate=args.min_success_rate,
    )
    recovery_path, recovery_passed = _write_recovery_report(
        args.recovery_output,
        metadata=run_metadata,
        attempts=attempt_results,
    )

    print(f"Wrote {reliability_path}")
    print(f"Wrote {recovery_path}")

    if not reliability_passed or not recovery_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
