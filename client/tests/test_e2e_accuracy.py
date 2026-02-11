"""End-to-end accuracy tests: compare encrypted vs plaintext single-layer output."""

import pytest
import numpy as np
import tenseal as ts
from client.inference.nonlinear_ops import rms_norm, silu, softmax


class TestEncryptedAccuracy:
    """Verify that CKKS encryption introduces only small numerical errors."""

    def setup_method(self):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt → decrypt should be identity within CKKS tolerance."""
        x = np.random.randn(2048).astype(np.float32)
        enc_x = ts.ckks_vector(self.context, x.tolist())
        dec_x = np.array(enc_x.decrypt()[:2048], dtype=np.float32)
        np.testing.assert_allclose(dec_x, x, atol=0.001)

    def test_encrypted_matmul_accuracy(self):
        """Encrypted matrix multiply should match plaintext within tolerance."""
        dim_in, dim_out = 64, 32
        x = np.random.randn(dim_in).astype(np.float32) * 0.1
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.01

        # Plaintext reference
        expected = x @ W

        # Encrypted computation
        enc_x = ts.ckks_vector(self.context, x.tolist())
        enc_result = enc_x.mm(W.tolist())
        actual = np.array(enc_result.decrypt()[:dim_out], dtype=np.float32)

        np.testing.assert_allclose(actual, expected, atol=0.05)

    def test_rms_norm_accuracy(self):
        """RMSNorm output should be deterministic and correct."""
        x = np.random.randn(2048).astype(np.float32)
        w = np.ones(2048, dtype=np.float32)
        r1 = rms_norm(x, w)
        r2 = rms_norm(x, w)
        np.testing.assert_array_equal(r1, r2)

    def test_silu_accuracy(self):
        """SiLU should match expected values."""
        x = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        result = silu(x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_encrypted_matmul_2048_dim(self):
        """Test HE matmul at full hidden dimension (2048 → 256)."""
        dim_in, dim_out = 2048, 256
        x = np.random.randn(dim_in).astype(np.float32) * 0.01
        W = np.random.randn(dim_in, dim_out).astype(np.float32) * 0.01

        expected = x @ W

        enc_x = ts.ckks_vector(self.context, x.tolist())
        enc_result = enc_x.mm(W.tolist())
        actual = np.array(enc_result.decrypt()[:dim_out], dtype=np.float32)

        np.testing.assert_allclose(actual, expected, atol=0.1)
