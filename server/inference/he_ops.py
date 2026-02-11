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


def he_matmul(enc_vector: ts.CKKSVector, weight_matrix: np.ndarray) -> ts.CKKSVector:
    """Multiply encrypted vector by plaintext weight matrix.

    enc_vector: encrypted vector of dim D_in
    weight_matrix: plaintext (D_in, D_out) where D_out <= SLOT_COUNT

    Returns encrypted vector of dim D_out.
    Uses TenSEAL's built-in matrix-vector multiplication (enc.mm).
    """
    # TenSEAL mm expects the weight as a list of lists
    # enc_vector.mm(matrix) computes: enc_vector @ matrix
    weight_list = weight_matrix.tolist()
    return enc_vector.mm(weight_list)


def he_matmul_split_output(
    enc_vector: ts.CKKSVector, weight_matrix: np.ndarray
) -> list[ts.CKKSVector]:
    """Matrix multiply when output dim > SLOT_COUNT.

    Splits the weight matrix columns into chunks that fit in SLOT_COUNT.
    Used for gate_proj and up_proj (2048 → 5632).

    Returns list of encrypted vectors, one per chunk.
    """
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
    enc_vector: ts.CKKSVector, weights: dict
) -> dict[str, ts.CKKSVector]:
    """Compute Q, K, V projections homomorphically.

    enc_vector: encrypted normed hidden state (dim 2048)
    weights: dict with q_proj, k_proj, v_proj matrices

    Returns dict with encrypted Q (2048), K (256), V (256).
    """
    enc_q = he_matmul(enc_vector, weights["q_proj"])
    enc_k = he_matmul(enc_vector, weights["k_proj"])
    enc_v = he_matmul(enc_vector, weights["v_proj"])
    return {"q": enc_q, "k": enc_k, "v": enc_v}


def compute_o_projection(
    enc_attn: ts.CKKSVector, weights: dict
) -> ts.CKKSVector:
    """Compute output projection: Enc(attn) @ W_o."""
    return he_matmul(enc_attn, weights["o_proj"])


def compute_ffn_gate_up(
    enc_vector: ts.CKKSVector, weights: dict
) -> dict[str, list[ts.CKKSVector]]:
    """Compute gate and up projections (2048 → 5632, split output).

    Returns dict with lists of encrypted chunks for gate and up.
    """
    gate_parts = he_matmul_split_output(enc_vector, weights["gate_proj"])
    up_parts = he_matmul_split_output(enc_vector, weights["up_proj"])
    return {"gate_parts": gate_parts, "up_parts": up_parts}


def compute_ffn_down(
    enc_vectors: list[ts.CKKSVector],
    weights: dict,
    chunk_sizes: list[int],
) -> ts.CKKSVector:
    """Compute down projection (5632 → 2048, split input)."""
    return he_matmul_split_input(enc_vectors, weights["down_proj"], chunk_sizes)
