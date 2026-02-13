"""Client-side non-linear operations for split inference.

These operations cannot be done homomorphically and must run on the client
where data is decrypted. They match PyTorch's LlamaRMSNorm, SiLU, and attention.
"""

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm matching PyTorch's LlamaRMSNorm.

    x: (..., hidden_dim)
    weight: (hidden_dim,)
    """
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x * (1.0 / np.sqrt(variance + eps))
    return x_normed * weight


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def compute_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 64,
) -> np.ndarray:
    """Multi-head attention with Grouped Query Attention (GQA).

    TinyLlama: 32 query heads, 4 KV heads (each KV head shared by 8 query heads).

    q: (seq_len, num_heads * head_dim) = (seq_len, 2048)
    k: (seq_len, num_kv_heads * head_dim) = (seq_len, 256)
    v: (seq_len, num_kv_heads * head_dim) = (seq_len, 256)

    Returns: (seq_len, num_heads * head_dim) = (seq_len, 2048)
    """
    seq_len = q.shape[0]
    kv_group_size = num_heads // num_kv_heads  # 8

    # Reshape to (num_heads, seq_len, head_dim)
    q = q.reshape(seq_len, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, num_kv_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, num_kv_heads, head_dim).transpose(1, 0, 2)

    # Repeat KV heads for GQA: (num_kv_heads, ...) -> (num_heads, ...)
    k = np.repeat(k, kv_group_size, axis=0)
    v = np.repeat(v, kv_group_size, axis=0)

    # Scaled dot-product attention
    scale = 1.0 / np.sqrt(head_dim)
    # (num_heads, seq_len, seq_len)
    attn_weights = np.matmul(q, k.transpose(0, 2, 1)) * scale

    # Causal mask: mask future positions
    if seq_len > 1:
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        attn_weights = attn_weights + mask[np.newaxis, :, :]

    attn_weights = softmax(attn_weights, axis=-1)

    # (num_heads, seq_len, head_dim)
    attn_output = np.matmul(attn_weights, v)

    # Reshape back to (seq_len, num_heads * head_dim)
    attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, num_heads * head_dim)
    return attn_output


def apply_rotary_embeddings(
    q: np.ndarray,
    k: np.ndarray,
    seq_len: int,
    head_dim: int = 64,
    num_heads: int = 32,
    num_kv_heads: int = 4,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Rotary Position Embeddings (RoPE) to Q and K.

    q: (seq_len, num_heads * head_dim)
    k: (seq_len, num_kv_heads * head_dim)
    """
    # Build frequency table
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # (seq_len, head_dim//2)

    cos = np.cos(freqs)  # (seq_len, head_dim//2)
    sin = np.sin(freqs)  # (seq_len, head_dim//2)

    def rotate_half(x_heads):
        """Apply RoPE to a set of heads.
        x_heads: (seq_len, n_heads, head_dim)
        """
        x1 = x_heads[..., : head_dim // 2]
        x2 = x_heads[..., head_dim // 2 :]
        # cos/sin are (seq_len, head_dim//2), broadcast over heads
        c = cos[:, np.newaxis, :]  # (seq_len, 1, head_dim//2)
        s = sin[:, np.newaxis, :]
        rotated = np.concatenate(
            [x1 * c - x2 * s, x1 * s + x2 * c], axis=-1
        )
        return rotated

    # Reshape to (seq_len, n_heads, head_dim) for rotation
    q_r = q.reshape(seq_len, num_heads, head_dim)
    k_r = k.reshape(seq_len, num_kv_heads, head_dim)

    q_rotated = rotate_half(q_r).reshape(seq_len, num_heads * head_dim)
    k_rotated = rotate_half(k_r).reshape(seq_len, num_kv_heads * head_dim)

    return q_rotated, k_rotated


def apply_rotary_embeddings_at_positions(
    q: np.ndarray,
    k: np.ndarray,
    positions: np.ndarray,
    head_dim: int = 64,
    num_heads: int = 32,
    num_kv_heads: int = 4,
    base: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply RoPE at explicit positions (supports KV-cached incremental inference).

    q: (seq_len, num_heads * head_dim)
    k: (seq_len, num_kv_heads * head_dim)
    positions: (seq_len,) integer position indices
    """
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.outer(positions.astype(np.float32), inv_freq)  # (seq_len, head_dim//2)

    cos = np.cos(freqs)
    sin = np.sin(freqs)

    def rotate_half(x_heads):
        x1 = x_heads[..., : head_dim // 2]
        x2 = x_heads[..., head_dim // 2 :]
        c = cos[:, np.newaxis, :]
        s = sin[:, np.newaxis, :]
        return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)

    seq_len = q.shape[0]
    q_r = q.reshape(seq_len, num_heads, head_dim)
    k_r = k.reshape(seq_len, num_kv_heads, head_dim)

    q_rotated = rotate_half(q_r).reshape(seq_len, num_heads * head_dim)
    k_rotated = rotate_half(k_r).reshape(seq_len, num_kv_heads * head_dim)

    return q_rotated, k_rotated


def compute_attention_cached(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    num_heads: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 64,
) -> np.ndarray:
    """Multi-head attention supporting KV cache (q_len may differ from kv_len).

    For initial pass: q_len == kv_len, applies causal mask.
    For incremental pass: q_len == 1, no mask needed (single token sees all past).

    q: (q_len, num_heads * head_dim)
    k: (kv_len, num_kv_heads * head_dim)
    v: (kv_len, num_kv_heads * head_dim)

    Returns: (q_len, num_heads * head_dim)
    """
    q_len = q.shape[0]
    kv_len = k.shape[0]
    kv_group_size = num_heads // num_kv_heads

    q_h = q.reshape(q_len, num_heads, head_dim).transpose(1, 0, 2)
    k_h = k.reshape(kv_len, num_kv_heads, head_dim).transpose(1, 0, 2)
    v_h = v.reshape(kv_len, num_kv_heads, head_dim).transpose(1, 0, 2)

    k_h = np.repeat(k_h, kv_group_size, axis=0)
    v_h = np.repeat(v_h, kv_group_size, axis=0)

    scale = 1.0 / np.sqrt(head_dim)
    attn_weights = np.matmul(q_h, k_h.transpose(0, 2, 1)) * scale

    # Causal mask (only needed when q_len > 1)
    if q_len > 1:
        diag_offset = kv_len - q_len + 1
        mask = np.triu(np.full((q_len, kv_len), -np.inf), k=diag_offset)
        attn_weights = attn_weights + mask[np.newaxis, :, :]

    attn_weights = softmax(attn_weights, axis=-1)
    attn_output = np.matmul(attn_weights, v_h)

    return attn_output.transpose(1, 0, 2).reshape(q_len, num_heads * head_dim)
