"""Probe helpers for OpenFHE backend readiness and smoke operations."""

from __future__ import annotations

import os
from typing import Any

from common import he_backend


def probe_openfhe_backend() -> dict[str, Any]:
    status = he_backend.get_backend_status()
    result: dict[str, Any] = {
        "selected_backend": status.get("selected_backend"),
        "openfhe_available": status.get("openfhe_available", False),
        "openfhe_import_error": status.get("openfhe_import_error"),
        "probe_passed": False,
        "steps": {
            "create_context": False,
            "serialize_public_context": False,
            "context_from_public_bytes": False,
            "encrypt_decrypt_roundtrip": False,
            "square": False,
            "matmul": False,
        },
        "error": None,
    }

    if not status.get("openfhe_available", False):
        result["error"] = "OpenFHE module import unavailable in this environment."
        return result

    prior_backend = os.environ.get("ZKLLM_HE_BACKEND")
    os.environ["ZKLLM_HE_BACKEND"] = "openfhe"
    try:
        context = he_backend.create_context(
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            global_scale=2**40,
            use_galois_keys=True,
            use_relin_keys=True,
        )
        result["steps"]["create_context"] = True

        public_raw = he_backend.serialize_public_context(context)
        result["steps"]["serialize_public_context"] = True

        public_context = he_backend.context_from_public_bytes(public_raw)
        result["steps"]["context_from_public_bytes"] = True

        values = [1.0, 2.0, 3.0]
        encrypted = he_backend.encrypt_vector(context, values)
        encrypted_public = he_backend.vector_from_bytes(
            public_context,
            he_backend.serialize_vector(encrypted),
        )

        private_for_decrypt = dict(encrypted_public)
        private_for_decrypt["secret_key"] = context.get("secret_key")
        private_for_decrypt["length"] = len(values)

        decrypted = he_backend.decrypt_vector(private_for_decrypt)
        result["steps"]["encrypt_decrypt_roundtrip"] = len(decrypted) >= len(values)

        squared = he_backend.square(encrypted)
        result["steps"]["square"] = bool(squared)

        matrix = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        multiplied = he_backend.matmul(encrypted, matrix)
        decrypted_mul = he_backend.decrypt_vector(multiplied)
        result["steps"]["matmul"] = len(decrypted_mul) >= 2
    except Exception as exc:  # pragma: no cover - depends on runtime backend
        result["error"] = str(exc)
    finally:
        if prior_backend is None:
            os.environ.pop("ZKLLM_HE_BACKEND", None)
        else:
            os.environ["ZKLLM_HE_BACKEND"] = prior_backend

    result["probe_passed"] = all(result["steps"].values())
    return result
