"""Tests for client-side nonlinear operations against PyTorch reference."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from client.inference.nonlinear_ops import rms_norm, silu, softmax, compute_attention


class TestRMSNorm:
    def test_matches_pytorch(self):
        """RMSNorm output should match LlamaRMSNorm within float32 tolerance."""
        hidden_dim = 2048
        eps = 1e-5
        x_np = np.random.randn(3, hidden_dim).astype(np.float32)
        weight_np = np.random.randn(hidden_dim).astype(np.float32)

        # Our implementation
        result = rms_norm(x_np, weight_np, eps)

        # PyTorch reference
        x_t = torch.tensor(x_np)
        weight_t = torch.tensor(weight_np)
        variance = x_t.pow(2).mean(-1, keepdim=True)
        ref = (x_t * torch.rsqrt(variance + eps)) * weight_t
        ref_np = ref.numpy()

        np.testing.assert_allclose(result, ref_np, rtol=1e-5, atol=1e-5)

    def test_single_vector(self):
        result = rms_norm(np.ones(64, dtype=np.float32), np.ones(64, dtype=np.float32))
        assert result.shape == (64,)
        np.testing.assert_allclose(result, np.ones(64), atol=1e-5)


class TestSiLU:
    def test_matches_pytorch(self):
        x_np = np.random.randn(100).astype(np.float32)
        result = silu(x_np)
        ref = torch.nn.functional.silu(torch.tensor(x_np)).numpy()
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    def test_zero(self):
        assert silu(np.array([0.0]))[0] == pytest.approx(0.0)


class TestSoftmax:
    def test_matches_pytorch(self):
        x_np = np.random.randn(5, 10).astype(np.float32)
        result = softmax(x_np, axis=-1)
        ref = torch.nn.functional.softmax(torch.tensor(x_np), dim=-1).numpy()
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    def test_sums_to_one(self):
        x = np.random.randn(4, 8).astype(np.float32)
        result = softmax(x, axis=-1)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-6)

    def test_numerical_stability(self):
        """Large values should not cause overflow."""
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        result = softmax(x, axis=-1)
        assert np.all(np.isfinite(result))
        assert result.sum() == pytest.approx(1.0, abs=1e-6)


class TestAttention:
    def test_output_shape(self):
        seq_len = 4
        hidden_dim = 2048
        num_kv_heads = 4
        head_dim = 64
        q = np.random.randn(seq_len, hidden_dim).astype(np.float32)
        k = np.random.randn(seq_len, num_kv_heads * head_dim).astype(np.float32)
        v = np.random.randn(seq_len, num_kv_heads * head_dim).astype(np.float32)
        result = compute_attention(q, k, v)
        assert result.shape == (seq_len, hidden_dim)

    def test_single_token(self):
        """Single token attention should just be V projected."""
        hidden_dim = 2048
        num_kv_heads = 4
        head_dim = 64
        q = np.random.randn(1, hidden_dim).astype(np.float32)
        k = np.random.randn(1, num_kv_heads * head_dim).astype(np.float32)
        v = np.random.randn(1, num_kv_heads * head_dim).astype(np.float32)
        result = compute_attention(q, k, v)
        assert result.shape == (1, hidden_dim)
        assert np.all(np.isfinite(result))
