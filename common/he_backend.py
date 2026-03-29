"""Backend abstraction for HE context and vector operations."""

from __future__ import annotations

import os
from typing import Any

import tenseal as ts

HEContext = Any
HEVector = Any

OPENFHE_IMPORT_ERROR: str | None = None
try:
    import openfhe as _openfhe  # noqa: F401
    OPENFHE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-specific
    OPENFHE_AVAILABLE = False
    OPENFHE_IMPORT_ERROR = str(exc)


def get_backend_name() -> str:
    return os.getenv("ZKLLM_HE_BACKEND", "tenseal").lower()


def get_backend_status() -> dict[str, Any]:
    return {
        "selected_backend": get_backend_name(),
        "openfhe_available": OPENFHE_AVAILABLE,
        "openfhe_import_error": OPENFHE_IMPORT_ERROR,
    }


def create_context(
    *,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: list[int],
    global_scale: int,
    use_galois_keys: bool,
    use_relin_keys: bool,
) -> HEContext:
    backend = get_backend_name()
    if backend != "tenseal":
        raise RuntimeError(
            f"Unsupported HE backend '{backend}'. OpenFHE is not ready in this environment: {OPENFHE_IMPORT_ERROR}"
        )

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.global_scale = global_scale

    if use_galois_keys:
        context.generate_galois_keys()
    if use_relin_keys:
        context.generate_relin_keys()

    return context


def context_from_public_bytes(ctx_bytes: bytes) -> HEContext:
    backend = get_backend_name()
    if backend != "tenseal":
        raise RuntimeError(
            f"Unsupported HE backend '{backend}'. OpenFHE is not ready in this environment: {OPENFHE_IMPORT_ERROR}"
        )
    return ts.context_from(ctx_bytes)


def serialize_public_context(context: HEContext) -> bytes:
    return context.serialize(
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )


def encrypt_vector(context: HEContext, values) -> HEVector:
    return ts.ckks_vector(context, list(values))


def decrypt_vector(vector: HEVector) -> list[float]:
    return vector.decrypt()


def serialize_vector(vector: HEVector) -> bytes:
    return vector.serialize()


def vector_from_bytes(context: HEContext, raw: bytes) -> HEVector:
    return ts.ckks_vector_from(context, raw)


def clone_vector(vector: HEVector) -> HEVector:
    return vector_from_bytes(vector.context(), serialize_vector(vector))


def matmul(vector: HEVector, matrix) -> HEVector:
    if isinstance(matrix, list):
        return vector.mm(matrix)
    return vector.mm(matrix.tolist())


def square(vector: HEVector) -> HEVector:
    return vector.square()
