"""Homomorphic encryption operations for server-side inference."""

from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import tenseal as ts
from common.constants import SLOT_COUNT
from common.logging_utils import get_logger

logger = get_logger("server.he_ops")
HE_POLY_SILU_COEFFS = (
    0.23970363,
    0.5,
    0.10245962,
)

# Reuse thread pool across calls (SEAL releases GIL for C++ operations)
_he_thread_pool = ThreadPoolExecutor(max_workers=4)


def he_matmul(
    enc_vector: ts.CKKSVector,
    weight_matrix: np.ndarray = None,
    precomputed_list: list | None = None,
) -> ts.CKKSVector:
    """Multiply encrypted vector by plaintext weight matrix.

    enc_vector: encrypted vector of dim D_in
    weight_matrix: plaintext (D_in, D_out) where D_out <= SLOT_COUNT
    precomputed_list: pre-converted .tolist() result (avoids repeated conversion)

    Returns encrypted vector of dim D_out.
    """
    t0 = time.perf_counter()
    if precomputed_list is not None:
        result = enc_vector.mm(precomputed_list)
    elif weight_matrix is not None:
        result = enc_vector.mm(weight_matrix.tolist())
    else:
        raise ValueError("Either weight_matrix or precomputed_list must be provided")
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(f"he_matmul: {elapsed:.1f}ms")
    return result


def he_matmul_split_output(
    enc_vector: ts.CKKSVector,
    weight_matrix: np.ndarray,
    precomputed_chunks: list | None = None,
) -> list[ts.CKKSVector]:
    """Matrix multiply when output dim > SLOT_COUNT.

    Splits the weight matrix columns into chunks that fit in SLOT_COUNT.
    Used for gate_proj and up_proj (2048 → 5632).

    Returns list of encrypted vectors, one per chunk.
    """
    if precomputed_chunks is not None:
        # Pre-split chunk lists provided — validate chunk count matches expected splits
        if not isinstance(precomputed_chunks, list) or len(precomputed_chunks) == 0:
            raise ValueError("precomputed_chunks must be a non-empty list")
        return [enc_vector.mm(chunk) for chunk in precomputed_chunks]

    d_in, d_out = weight_matrix.shape
    if d_out <= SLOT_COUNT:
        return [he_matmul(enc_vector, weight_matrix)]

    results = []
    for start in range(0, d_out, SLOT_COUNT):
        end = min(start + SLOT_COUNT, d_out)
        chunk = weight_matrix[:, start:end]
        results.append(he_matmul(enc_vector, chunk))
    return results


def he_matmul_split_input(
    enc_vectors: list[ts.CKKSVector],
    weight_matrix: np.ndarray,
    chunk_sizes: list[int],
) -> ts.CKKSVector:
    """Matrix multiply when input dim > SLOT_COUNT.

    Splits the weight matrix rows to match the encrypted input chunks,
    multiplies each chunk, and sums the results.
    Used for down_proj (5632 → 2048).

    enc_vectors: list of encrypted chunks
    weight_matrix: (D_in, D_out) full weight matrix
    chunk_sizes: size of each input chunk

    Returns single encrypted vector of dim D_out.
    """
    result = None
    row_start = 0
    for enc_chunk, size in zip(enc_vectors, chunk_sizes):
        w_chunk = weight_matrix[row_start : row_start + size, :]
        partial = he_matmul(enc_chunk, w_chunk)
        if result is None:
            result = partial
        else:
            result = result + partial
        row_start += size
    return result


def compute_qkv_projections(
    enc_vector: ts.CKKSVector, weights: dict, weight_lists: dict | None = None,
) -> dict[str, ts.CKKSVector]:
    """Compute Q, K, V projections homomorphically.

    enc_vector: encrypted normed hidden state (dim 2048)
    weights: dict with q_proj, k_proj, v_proj matrices
    weight_lists: optional pre-converted .tolist() dict for avoiding repeated conversion

    Returns dict with encrypted Q (2048), K (256), V (256).
    """
    wl = weight_lists or {}

    # Create independent copies of enc_vector for thread safety.
    # TenSEAL/SEAL does not guarantee thread-safe concurrent reads on the
    # same CKKSVector object (internal evaluator may use shared buffers).
    raw = enc_vector.serialize()
    ctx = enc_vector.context()
    enc_q = ts.ckks_vector_from(ctx, raw)
    enc_k = ts.ckks_vector_from(ctx, raw)
    enc_v = ts.ckks_vector_from(ctx, raw)

    # Submit Q, K, V matmuls in parallel (they're independent)
    q_future = _he_thread_pool.submit(
        he_matmul, enc_q, weights["q_proj"], precomputed_list=wl.get("q_proj"))
    k_future = _he_thread_pool.submit(
        he_matmul, enc_k, weights["k_proj"], precomputed_list=wl.get("k_proj"))
    v_future = _he_thread_pool.submit(
        he_matmul, enc_v, weights["v_proj"], precomputed_list=wl.get("v_proj"))

    return {"q": q_future.result(), "k": k_future.result(), "v": v_future.result()}


def compute_o_projection(
    enc_attn: ts.CKKSVector, weights: dict, weight_lists: dict | None = None,
) -> ts.CKKSVector:
    """Compute output projection: Enc(attn) @ W_o."""
    wl = weight_lists or {}
    return he_matmul(enc_attn, weights["o_proj"],
                     precomputed_list=wl.get("o_proj"))


def compute_ffn_gate_up(
    enc_vector: ts.CKKSVector, weights: dict, weight_lists: dict | None = None,
) -> dict[str, list[ts.CKKSVector]]:
    """Compute gate and up projections (2048 → 5632, split output).

    Returns dict with lists of encrypted chunks for gate and up.
    """
    wl = weight_lists or {}
    gate_chunks = wl.get("gate_proj")
    up_chunks = wl.get("up_proj")

    # Check if pre-split chunks (list of list-of-lists) are available
    if gate_chunks is not None and isinstance(gate_chunks, list) and len(gate_chunks) > 0 and isinstance(gate_chunks[0], list) and len(gate_chunks[0]) > 0 and isinstance(gate_chunks[0][0], list):
        # Create independent copies for thread safety (avoid concurrent access to same CKKSVector)
        raw = enc_vector.serialize()
        ctx = enc_vector.context()
        num_copies = len(gate_chunks) + len(up_chunks)
        copies = [ts.ckks_vector_from(ctx, raw) for _ in range(num_copies)]
        # Submit all chunk matmuls in parallel
        gate_futures = [_he_thread_pool.submit(copies[i].mm, chunk) for i, chunk in enumerate(gate_chunks)]
        up_futures = [_he_thread_pool.submit(copies[len(gate_chunks) + i].mm, chunk) for i, chunk in enumerate(up_chunks)]
        gate_parts = [f.result() for f in gate_futures]
        up_parts = [f.result() for f in up_futures]
    else:
        # Create copies for thread safety
        raw = enc_vector.serialize()
        ctx = enc_vector.context()
        enc_gate = ts.ckks_vector_from(ctx, raw)
        enc_up = ts.ckks_vector_from(ctx, raw)
        gate_future = _he_thread_pool.submit(he_matmul_split_output, enc_gate, weights["gate_proj"])
        up_future = _he_thread_pool.submit(he_matmul_split_output, enc_up, weights["up_proj"])
        gate_parts = gate_future.result()
        up_parts = up_future.result()
    return {"gate_parts": gate_parts, "up_parts": up_parts}


def compute_ffn_down(
    enc_vectors: list[ts.CKKSVector],
    weights: dict,
    chunk_sizes: list[int],
) -> ts.CKKSVector:
    """Compute down projection (5632 → 2048, split input)."""
    return he_matmul_split_input(enc_vectors, weights["down_proj"], chunk_sizes)


def poly_silu(
    enc_vec: ts.CKKSVector,
    coeffs: tuple[float, ...] = HE_POLY_SILU_COEFFS,
) -> ts.CKKSVector:
    """Evaluate the HE-safe SiLU polynomial approximation on an encrypted vector."""
    x2 = enc_vec.square()

    result = enc_vec * coeffs[1]
    result += coeffs[0]
    result += x2 * coeffs[2]
    return result


def compute_ffn_merged(
    enc_vector: ts.CKKSVector,
    weights: dict,
    chunk_sizes: list[int],
    weight_lists: dict | None = None,
    coeffs: tuple[float, ...] = HE_POLY_SILU_COEFFS,
) -> ts.CKKSVector:
    """Compute gate -> poly SiLU -> multiply by up -> down projection in one server step."""
    gate_up = compute_ffn_gate_up(enc_vector, weights, weight_lists=weight_lists)
    activated_chunks = [poly_silu(part, coeffs=coeffs) for part in gate_up["gate_parts"]]
    multiplied_chunks = [
        activated * up
        for activated, up in zip(activated_chunks, gate_up["up_parts"])
    ]
    return compute_ffn_down(multiplied_chunks, weights, chunk_sizes)
