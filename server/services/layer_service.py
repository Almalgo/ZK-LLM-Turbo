"""Layer inference service functions shared by HTTP and HaaS adapters."""

from __future__ import annotations

import base64
import time

import msgpack
import zstandard as zstd
from fastapi import HTTPException

from common.he_backend import serialize_vector, vector_from_bytes
from common.logging_utils import get_logger
from server.inference.he_ops import (
    compute_ffn_down,
    compute_ffn_gate_up,
    compute_ffn_merged,
    compute_o_projection,
    compute_qkv_projections,
)
from server.model.weight_manager import get_layer_weight_lists, get_layer_weights
from server.services.session_service import get_session

logger = get_logger("server.layer_service")

_zstd_compressor = zstd.ZstdCompressor(level=3)
_zstd_decompressor = zstd.ZstdDecompressor()


def deserialize_vectors(context, vectors_b64: list[str]) -> list[object]:
    """Deserialize base64-encoded encrypted vectors using the session's public context."""
    result = []
    for b64 in vectors_b64:
        raw = base64.b64decode(b64)
        vec = vector_from_bytes(context, raw)
        result.append(vec)
    return result


def serialize_vectors(vectors: list[object]) -> list[str]:
    """Serialize encrypted vectors to base64."""
    return [base64.b64encode(serialize_vector(v)).decode("utf-8") for v in vectors]


def dispatch_operation(
    operation: str,
    enc_vectors: list[object],
    weights: dict,
    weight_lists: dict,
    chunk_sizes: list[int] | None,
) -> list[object]:
    if operation == "qkv":
        result_vectors = []
        for enc_v in enc_vectors:
            qkv = compute_qkv_projections(enc_v, weights, weight_lists=weight_lists)
            result_vectors.extend([qkv["q"], qkv["k"], qkv["v"]])
        return result_vectors

    if operation == "o_proj":
        return [compute_o_projection(enc_v, weights, weight_lists=weight_lists) for enc_v in enc_vectors]

    if operation == "ffn_gate_up":
        result_vectors = []
        for enc_v in enc_vectors:
            gu = compute_ffn_gate_up(enc_v, weights, weight_lists=weight_lists)
            result_vectors.extend(gu["gate_parts"] + gu["up_parts"])
        return result_vectors

    if operation == "ffn_down":
        if chunk_sizes is None:
            raise HTTPException(status_code=400, detail="ffn_down requires chunk_sizes")
        num_chunks = len(chunk_sizes)
        batch_size = len(enc_vectors) // num_chunks
        result_vectors = []
        for i in range(batch_size):
            token_chunks = enc_vectors[i * num_chunks : (i + 1) * num_chunks]
            down = compute_ffn_down(token_chunks, weights, chunk_sizes)
            result_vectors.append(down)
        return result_vectors

    if operation == "ffn_merged":
        if chunk_sizes is None:
            raise HTTPException(status_code=400, detail="ffn_merged requires chunk_sizes")
        return [
            compute_ffn_merged(
                enc_v,
                weights,
                chunk_sizes,
                weight_lists=weight_lists,
            )
            for enc_v in enc_vectors
        ]

    raise HTTPException(status_code=400, detail=f"Unknown operation: {operation}")


def process_layer_request(req_data: dict, cid: str | None = None) -> dict:
    """Process one JSON-compatible layer operation."""
    start = time.perf_counter()

    session_id = req_data["session_id"]
    layer_idx = req_data["layer_idx"]
    operation = req_data["operation"]
    encrypted_vectors_b64 = req_data["encrypted_vectors_b64"]
    chunk_sizes = req_data.get("chunk_sizes")
    pack_counts = req_data.get("pack_counts")

    context = get_session(session_id)
    weights = get_layer_weights(layer_idx)
    weight_lists = get_layer_weight_lists(layer_idx)
    enc_vectors = deserialize_vectors(context, encrypted_vectors_b64)

    logger.info(
        "Processing layer op",
        extra={"extra": {
            "cid": cid,
            "layer": layer_idx,
            "op": operation,
            "num_vectors": len(enc_vectors),
            "pack_counts": pack_counts,
        }},
    )

    result_vectors = dispatch_operation(
        operation,
        enc_vectors,
        weights,
        weight_lists,
        chunk_sizes,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Layer op complete",
        extra={"extra": {
            "cid": cid,
            "op": operation,
            "elapsed_ms": round(elapsed_ms, 2),
        }},
    )

    return {
        "encrypted_results_b64": serialize_vectors(result_vectors),
        "operation": operation,
        "layer_idx": layer_idx,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def process_binary_payload(req_data: dict, cid: str) -> bytes:
    """Process one msgpack binary layer operation."""
    start = time.perf_counter()

    session_id = req_data["session_id"]
    layer_idx = req_data["layer_idx"]
    operation = req_data["operation"]
    encrypted_vectors_raw = req_data["encrypted_vectors"]
    chunk_sizes = req_data.get("chunk_sizes")
    pack_counts = req_data.get("pack_counts")

    if not isinstance(layer_idx, int) or layer_idx < 0:
        raise HTTPException(status_code=400, detail="Invalid layer_idx: must be a non-negative integer")
    if not isinstance(session_id, str) or len(session_id) > 256:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not isinstance(encrypted_vectors_raw, list):
        raise HTTPException(status_code=400, detail="encrypted_vectors must be a list")
    if len(encrypted_vectors_raw) > 100:
        raise HTTPException(status_code=400, detail="Too many vectors (max 100)")

    context = get_session(session_id)
    weights = get_layer_weights(layer_idx)
    weight_lists = get_layer_weight_lists(layer_idx)

    max_decompressed_size = 50_000_000
    enc_vectors = [
        vector_from_bytes(context, _zstd_decompressor.decompress(raw, max_output_size=max_decompressed_size))
        for raw in encrypted_vectors_raw
    ]

    logger.info(
        "Processing layer op (binary)",
        extra={"extra": {
            "cid": cid,
            "layer": layer_idx,
            "op": operation,
            "num_vectors": len(enc_vectors),
            "pack_counts": pack_counts,
        }},
    )

    result_vectors = dispatch_operation(operation, enc_vectors, weights, weight_lists, chunk_sizes)
    results_raw = [_zstd_compressor.compress(serialize_vector(v)) for v in result_vectors]
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Layer op complete (binary)",
        extra={"extra": {"cid": cid, "op": operation, "elapsed_ms": round(elapsed_ms, 2)}},
    )

    return msgpack.packb({
        "encrypted_results": results_raw,
        "operation": operation,
        "layer_idx": layer_idx,
        "elapsed_ms": round(elapsed_ms, 2),
    }, use_bin_type=True)
