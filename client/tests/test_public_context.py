"""Tests for public context sharing — verify server cannot decrypt."""

import pytest
import numpy as np
import tenseal as ts
from client.encryption.ckks_context import create_ckks_context, serialize_public_context


class TestPublicContext:
    def test_public_context_has_no_secret_key(self):
        """Serialized public context should not contain secret key."""
        context = create_ckks_context()
        public_bytes = serialize_public_context(context)

        # Load as public context
        public_ctx = ts.context_from(public_bytes)

        # Encrypt with original (has secret key)
        vec = ts.ckks_vector(context, [1.0, 2.0, 3.0])
        serialized = vec.serialize()

        # Deserialize with public context
        pub_vec = ts.ckks_vector_from(public_ctx, serialized)

        # Server should NOT be able to decrypt
        with pytest.raises(Exception):
            pub_vec.decrypt()

    def test_server_can_compute_on_encrypted(self):
        """Public context should allow HE computation (matmul) but not decryption."""
        context = create_ckks_context()
        public_bytes = serialize_public_context(context)
        public_ctx = ts.context_from(public_bytes)

        # Client encrypts
        x = [1.0, 2.0, 3.0, 4.0]
        enc_x = ts.ckks_vector(context, x)
        enc_bytes = enc_x.serialize()

        # Server receives and computes
        server_vec = ts.ckks_vector_from(public_ctx, enc_bytes)
        # Plaintext-ciphertext multiplication
        result = server_vec * [2.0, 2.0, 2.0, 2.0]

        # Server sends result back
        result_bytes = result.serialize()

        # Client decrypts
        client_vec = ts.ckks_vector_from(context, result_bytes)
        decrypted = client_vec.decrypt()[:4]

        np.testing.assert_allclose(decrypted, [2.0, 4.0, 6.0, 8.0], atol=0.01)

    def test_server_matmul_works(self):
        """Server can do matrix-vector multiply with public context."""
        context = create_ckks_context()
        public_bytes = serialize_public_context(context)
        public_ctx = ts.context_from(public_bytes)

        # Client encrypts a small vector
        x = [1.0, 2.0, 3.0]
        enc_x = ts.ckks_vector(context, x)
        enc_bytes = enc_x.serialize()

        # Server computes x @ W where W is 3x2
        W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        server_vec = ts.ckks_vector_from(public_ctx, enc_bytes)
        result = server_vec.mm(W)
        result_bytes = result.serialize()

        # Client decrypts
        client_vec = ts.ckks_vector_from(context, result_bytes)
        decrypted = client_vec.decrypt()[:2]

        # Expected: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        np.testing.assert_allclose(decrypted, [4.0, 5.0], atol=0.01)
