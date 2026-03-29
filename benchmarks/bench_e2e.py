"""Benchmark end-to-end seconds-per-token across encrypted layer counts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from benchmarks.common import require_server, summarize_samples, write_benchmark_report
from client.client import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end encrypted generation latency.")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt used for generation.")
    parser.add_argument("--num-tokens", type=int, default=2, help="Tokens to generate per sample.")
    parser.add_argument("--samples", type=int, default=1, help="Number of runs per encrypted layer count.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=[1, 5, 11, 22],
        help="Encrypted layer counts to benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/bench_e2e.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()
    from client.client import load_config

    _, server_cfg = load_config()
    require_server(server_cfg["base_url"])

    results = []
    for encrypted_layers in args.layers:
        sample_values = []
        for _ in range(args.samples):
            run = generate(
                prompt=args.prompt,
                num_tokens=args.num_tokens,
                num_encrypted_layers=encrypted_layers,
                show_stats=False,
                return_stats=True,
                quiet=True,
            )
            ms_per_token = (run.stats["total"] / max(run.tokens_generated, 1)) * 1000
            sample_values.append(ms_per_token)

        results.append(
            summarize_samples(
                f"e2e_ms_per_token_layers_{encrypted_layers}",
                sample_values,
                metadata={"encrypted_layers": encrypted_layers, "num_tokens": args.num_tokens},
            )
        )

    output_path = write_benchmark_report(
        args.output,
        results,
        metadata={"prompt": args.prompt, "samples_per_layer": args.samples},
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
