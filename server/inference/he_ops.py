"""Homomorphic encryption operations for server-side inference.

Key constraint: poly_modulus_degree=8192 → 4096 CKKS slots.
Hidden dim (2048) fits in one ciphertext.
FFN intermediate dim (5632) does NOT fit → must split across 2 ciphertexts.
"""

import numpy as np
import tenseal as ts
from common.logging_utils import get_logger

logger = get_logger("server.he_ops")

SLOT_COUNT = 4096  # poly_modulus_degree // 2


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
    if precomputed_list is not None:
        return enc_vector.mm(precomputed_list)
    weight_list = weight_matrix.tolist()
    return enc_vector.mm(weight_list)


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
        # Pre-split chunk lists provided
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
    enc_q = he_matmul(enc_vector, weights["q_proj"],
                      precomputed_list=wl.get("q_proj"))
    enc_k = he_matmul(enc_vector, weights["k_proj"],
                      precomputed_list=wl.get("k_proj"))
    enc_v = he_matmul(enc_vector, weights["v_proj"],
                      precomputed_list=wl.get("v_proj"))
    return {"q": enc_q, "k": enc_k, "v": enc_v}


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
        gate_parts = he_matmul_split_output(enc_vector, None, precomputed_chunks=gate_chunks)
        up_parts = he_matmul_split_output(enc_vector, None, precomputed_chunks=up_chunks)
    else:
        gate_parts = he_matmul_split_output(enc_vector, weights["gate_proj"])
        up_parts = he_matmul_split_output(enc_vector, weights["up_proj"])
    return {"gate_parts": gate_parts, "up_parts": up_parts}


def compute_ffn_down(
    enc_vectors: list[ts.CKKSVector],
    weights: dict,
    chunk_sizes: list[int],
    weight_lists: dict | None = None,
) -> ts.CKKSVector:
    """Compute down projection (5632 → 2048, split input)."""
    return he_matmul_split_input(enc_vectors, weights["down_proj"], chunk_sizes)
