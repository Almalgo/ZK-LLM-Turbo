#!/usr/bin/env python3
"""Milestone 5 smoke test for SingularityNet daemon passthrough.

Validates a daemon-routed split-inference path by calling:
1) POST /api/session
2) POST /api/layer (operation=qkv)
"""

from __future__ import annotations

import argparse
import base64
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


def _write_artifact(
    output_path: Path,
    *,
    ok: bool,
    metadata: dict,
    error: str | None = None,
) -> Path:
    result = {
        "name": "m5_snet_smoke",
        "status": "pass" if ok else "fail",
        "metadata": metadata,
    }
    if error:
        result["error"] = error
    return write_benchmark_report(output_path, [result], metadata=metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 5 daemon passthrough smoke test")
    parser.add_argument("--base-url", required=True, help="Daemon base URL, e.g. http://host:7000")
    parser.add_argument("--session-endpoint", default="/api/session", help="Session endpoint path")
    parser.add_argument("--layer-endpoint", default="/api/layer", help="Layer endpoint path")
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index for smoke call")
    parser.add_argument("--operation", choices=["qkv"], default="qkv", help="Layer operation")
    parser.add_argument("--vector-size", type=int, default=2048, help="Input vector size for qkv")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds")
    parser.add_argument("--auth-token", default=None, help="Optional bearer token")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/m5_snet_smoke_sepolia.json"),
        help="Artifact output path",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    base_url = args.base_url.rstrip("/")
    session_url = f"{base_url}{args.session_endpoint}"
    layer_url = f"{base_url}{args.layer_endpoint}"
    headers = _headers(args.auth_token)
    metadata = {
        "run_utc": datetime.now(UTC).isoformat(),
        "daemon_base_url": base_url,
        "session_endpoint": args.session_endpoint,
        "layer_endpoint": args.layer_endpoint,
        "layer_idx": args.layer_idx,
        "operation": args.operation,
        "vector_size": args.vector_size,
        "seed": args.seed,
    }

    try:
        context = create_ckks_context()
        public_context_b64 = base64.b64encode(serialize_public_context(context)).decode("utf-8")

        t0 = time.perf_counter()
        session_resp = requests.post(
            session_url,
            json={"public_context_b64": public_context_b64},
            headers=headers,
            timeout=args.timeout,
        )
        session_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        session_resp.raise_for_status()
        session_json = session_resp.json()
        session_id = session_json.get("session_id")
        if not session_id:
            raise RuntimeError("Session response missing 'session_id'")

        rng = np.random.default_rng(args.seed)
        plaintext_vec = rng.normal(0.0, 0.01, size=args.vector_size).astype(np.float32)
        encrypted_vec = encrypt_vector(context, plaintext_vec.tolist())
        encrypted_vec_b64 = base64.b64encode(serialize_vector(encrypted_vec)).decode("utf-8")

        layer_payload = {
            "session_id": session_id,
            "layer_idx": args.layer_idx,
            "operation": args.operation,
            "encrypted_vectors_b64": [encrypted_vec_b64],
        }

        t0 = time.perf_counter()
        layer_resp = requests.post(
            layer_url,
            json=layer_payload,
            headers=headers,
            timeout=args.timeout,
        )
        layer_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        layer_resp.raise_for_status()
        layer_json = layer_resp.json()

        encrypted_results = layer_json.get("encrypted_results_b64")
        if not isinstance(encrypted_results, list):
            raise RuntimeError("Layer response missing list field 'encrypted_results_b64'")
        if len(encrypted_results) != 3:
            raise RuntimeError(
                f"Expected 3 encrypted results for qkv, got {len(encrypted_results)}"
            )

        metadata.update(
            {
                "session_status_code": session_resp.status_code,
                "layer_status_code": layer_resp.status_code,
                "session_id_present": True,
                "result_count": len(encrypted_results),
                "session_elapsed_ms": round(session_elapsed_ms, 3),
                "layer_elapsed_ms": round(layer_elapsed_ms, 3),
                "total_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "server_elapsed_ms": layer_json.get("elapsed_ms"),
            }
        )
        written = _write_artifact(args.output, ok=True, metadata=metadata)
        print(f"PASS wrote {written}")
        return 0

    except Exception as exc:
        metadata.update(
            {
                "total_elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        written = _write_artifact(args.output, ok=False, metadata=metadata, error=str(exc))
        print(f"FAIL wrote {written}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
