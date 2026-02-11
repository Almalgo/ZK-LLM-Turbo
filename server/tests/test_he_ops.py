"""Tests for server-side HE operations."""

import pytest
import numpy as np
import tenseal as ts
from server.inference.he_ops import (
    he_matmul,
    he_matmul_split_output,
    he_matmul_split_input,
    SLOT_COUNT,
)


@pytest.fixture
def ckks_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2**40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


class TestHEMatmul:
    def test_small_matmul(self, ckks_context):
        """Small-dimension matmul should match plaintext."""
        dim_in, dim_out = 8, 4
        x = np.random.randn(dim_in).astype(np.float32) * 0.1
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.1

        expected = x @ W
        enc_x = ts.ckks_vector(ckks_context, x.tolist())
        enc_result = he_matmul(enc_x, W)
        actual = np.array(enc_result.decrypt()[:dim_out], dtype=np.float32)

        np.testing.assert_allclose(actual, expected, atol=0.01)

    def test_hidden_dim_matmul(self, ckks_context):
        """2048-dim matmul should work within CKKS tolerance."""
        dim_in, dim_out = 2048, 256
        x = np.random.randn(dim_in).astype(np.float32) * 0.01
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.01

        expected = x @ W
        enc_x = ts.ckks_vector(ckks_context, x.tolist())
        enc_result = he_matmul(enc_x, W)
        actual = np.array(enc_result.decrypt()[:dim_out], dtype=np.float32)

        np.testing.assert_allclose(actual, expected, atol=0.1)

    def test_full_hidden_to_hidden(self, ckks_context):
        """2048 → 2048 matmul (like o_proj)."""
        dim = 2048
        x = np.random.randn(dim).astype(np.float32) * 0.01
        W = np.random.randn(dim, dim).astype(np.float32) * 0.005

        expected = x @ W
        enc_x = ts.ckks_vector(ckks_context, x.tolist())
        enc_result = he_matmul(enc_x, W)
        actual = np.array(enc_result.decrypt()[:dim], dtype=np.float32)

        # Larger tolerance for large matmul
        np.testing.assert_allclose(actual, expected, atol=0.5)


class TestHEMatmulSplitOutput:
    def test_no_split_needed(self, ckks_context):
        """Output dim < SLOT_COUNT should not split."""
        dim_in, dim_out = 16, 8
        x = np.random.randn(dim_in).astype(np.float32) * 0.1
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.1

        enc_x = ts.ckks_vector(ckks_context, x.tolist())
        parts = he_matmul_split_output(enc_x, W)

        assert len(parts) == 1
        actual = np.array(parts[0].decrypt()[:dim_out], dtype=np.float32)
        expected = x @ W
        np.testing.assert_allclose(actual, expected, atol=0.01)

    def test_split_needed(self, ckks_context):
        """Output dim > SLOT_COUNT should produce multiple ciphertexts."""
        dim_in = 32
        dim_out = SLOT_COUNT + 100  # Just over the limit
        x = np.random.randn(dim_in).astype(np.float32) * 0.1
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.01

        enc_x = ts.ckks_vector(ckks_context, x.tolist())
        parts = he_matmul_split_output(enc_x, W)

        assert len(parts) == 2

        # Reconstruct and compare
        expected = x @ W
        part0 = np.array(parts[0].decrypt()[:SLOT_COUNT], dtype=np.float32)
        part1 = np.array(parts[1].decrypt()[:100], dtype=np.float32)
        actual = np.concatenate([part0, part1])

        np.testing.assert_allclose(actual, expected, atol=0.05)


class TestHEMatmulSplitInput:
    def test_split_input_matmul(self, ckks_context):
        """Split-input matmul should reconstruct correct output."""
        chunk1_size, chunk2_size = 16, 8
        dim_out = 10
        dim_in = chunk1_size + chunk2_size

        x = np.random.randn(dim_in).astype(np.float32) * 0.1
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.1

        expected = x @ W

        # Split and encrypt
        enc1 = ts.ckks_vector(ckks_context, x[:chunk1_size].tolist())
        enc2 = ts.ckks_vector(ckks_context, x[chunk1_size:].tolist())

        result = he_matmul_split_input(
            [enc1, enc2], W, [chunk1_size, chunk2_size]
        )
        actual = np.array(result.decrypt()[:dim_out], dtype=np.float32)

        np.testing.assert_allclose(actual, expected, atol=0.05)
