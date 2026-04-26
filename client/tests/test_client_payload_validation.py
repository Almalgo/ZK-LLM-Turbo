import pytest
import numpy as np
from client.encryption.ckks_context import create_ckks_context
from client.inference.nonlinear_ops import rms_norm, silu, softmax


def test_rms_norm_input_validation():
    x = np.random.randn(2048).astype(np.float32)
    w = np.ones(2048, dtype=np.float32)
    result = rms_norm(x, w, eps=1e-5)
    assert result.shape == x.shape


def test_silu_input_validation():
    x = np.random.randn(1000).astype(np.float32)
    result = silu(x)
    assert result.shape == x.shape


def test_softmax_input_validation():
    x = np.random.randn(32, 64).astype(np.float32)
    result = softmax(x)
    assert result.shape == x.shape


def test_softmax_returns_valid_probabilities():
    x = np.random.randn(10, 20).astype(np.float32)
    result = softmax(x)
    assert np.all(result >= 0)
    assert np.all(result <= 1)
    row_sums = result.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)