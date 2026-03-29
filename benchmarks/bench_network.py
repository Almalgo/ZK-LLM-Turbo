"""Benchmark live encrypted round-trip latency against the running server."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import tenseal as ts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmarks.common import seeded_rng, summarize_samples, write_benchmark_report
from client.client import load_config, setup_session
from client.encryption.ckks_context import create_ckks_context
from client.inference.layer_protocol import EncryptedLayerProtocol


def _encrypt(context: ts.Context, vector: np.ndarray) -> ts.CKKSVector:
    return ts.ckks_vector(context, vector.tolist())


def _collect_metrics(protocol, layer_idx: int, operation: str, vectors, chunk_sizes=None):
    protocol.reset_round_metrics()
    protocol._send_request(layer_idx, operation, vectors, chunk_sizes=chunk_sizes)
    return protocol.get_round_metrics()[-1]


def _metric_metadata(samples: list[dict]) -> dict:
    keys = ["serialize_ms", "server_ms", "network_ms", "deserialize_ms", "payload_kb", "response_kb"]
    return {
        key: round(sum(sample[key] for sample in samples) / len(samples), 3)
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark network round trips against the inference server.")
    parser.add_argument("--samples", type=int, default=5, help="Number of measured samples per operation.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed.")
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index used for server-side weights.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/bench_network.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    _, server_cfg = load_config()
    context = create_ckks_context()
    session_id = setup_session(context, server_cfg)
    protocol = EncryptedLayerProtocol(
        context=context,
        session_id=session_id,
        server_url=server_cfg["base_url"],
        layer_endpoint=server_cfg["layer_endpoint"],
        auth_token=server_cfg["auth_token"],
        model_config=SimpleNamespace(),
    )
    rng = seeded_rng(args.seed)

    operations = {
        "qkv": lambda: [_encrypt(context, rng.normal(0.0, 0.01, size=2048).astype(np.float32))],
        "o_proj": lambda: [_encrypt(context, rng.normal(0.0, 0.01, size=2048).astype(np.float32))],
        "ffn_gate_up": lambda: [_encrypt(context, rng.normal(0.0, 0.01, size=2048).astype(np.float32))],
        "ffn_down": lambda: [
            _encrypt(context, rng.normal(0.0, 0.01, size=4096).astype(np.float32)),
            _encrypt(context, rng.normal(0.0, 0.01, size=1536).astype(np.float32)),
        ],
    }

    chunk_sizes = {"ffn_down": [4096, 1536]}
    results = []
    summary_metadata = {
        "server_base_url": server_cfg["base_url"],
        "layer_idx": args.layer_idx,
        "seed": args.seed,
        "operations": {},
    }

    for operation, vector_factory in operations.items():
        metrics = []
        for _ in range(args.samples):
            metrics.append(
                _collect_metrics(
                    protocol,
                    args.layer_idx,
                    operation,
                    vector_factory(),
                    chunk_sizes=chunk_sizes.get(operation),
                )
            )

        results.append(
            summarize_samples(
                f"{operation}_roundtrip",
                [metric["roundtrip_ms"] for metric in metrics],
                metadata=_metric_metadata(metrics),
            )
        )
        summary_metadata["operations"][operation] = _metric_metadata(metrics)

    output_path = write_benchmark_report(args.output, results, metadata=summary_metadata)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
