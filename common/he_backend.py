"""Backend abstraction for HE context and vector operations."""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Any

import msgpack
import numpy as np
import tenseal as ts

HEContext = Any
HEVector = Any

_OPENFHE_PLAINTEXT_CACHE: dict[tuple[int, int, int, int], list[Any]] = {}
_OPENFHE_PLAINTEXT_CACHE_LOCK = threading.Lock()

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
    gpu_available, gpu_error = probe_gpu_support()
    return {
        "selected_backend": get_backend_name(),
        "openfhe_available": OPENFHE_AVAILABLE,
        "openfhe_import_error": OPENFHE_IMPORT_ERROR,
        "gpu_available": gpu_available,
        "gpu_error": gpu_error,
    }


def probe_gpu_support() -> tuple[bool, str | None]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "nvidia-smi not found"

    if proc.returncode == 0:
        return True, None
    return False, (proc.stderr or proc.stdout or "GPU probe failed").strip()


def create_context(
    *,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: list[int],
    global_scale: int,
    use_galois_keys: bool,
    use_relin_keys: bool,
) -> HEContext:
    backend = get_backend_name()
    if backend == "tenseal":
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

    if backend == "openfhe":
        if not OPENFHE_AVAILABLE:
            raise RuntimeError(
                f"Unsupported HE backend '{backend}'. OpenFHE is not ready in this environment: {OPENFHE_IMPORT_ERROR}"
            )

        params = _openfhe.CCParamsCKKSRNS()
        params.SetRingDim(poly_modulus_degree)
        params.SetBatchSize(poly_modulus_degree // 2)
        if len(coeff_mod_bit_sizes) >= 2:
            params.SetScalingModSize(int(coeff_mod_bit_sizes[1]))
        multiplicative_depth = max(1, len(coeff_mod_bit_sizes) - 2)
        params.SetMultiplicativeDepth(multiplicative_depth)

        context = _openfhe.GenCryptoContext(params)
        context.Enable(_openfhe.PKE)
        context.Enable(_openfhe.KEYSWITCH)
        context.Enable(_openfhe.LEVELEDSHE)
        context.Enable(_openfhe.ADVANCEDSHE)

        keypair = context.KeyGen()
        if use_relin_keys:
            context.EvalMultKeyGen(keypair.secretKey)
        eval_sum_keys_generated = False
        if use_galois_keys:
            context.EvalRotateKeyGen(keypair.secretKey, [1, -1])
            context.EvalSumKeyGen(keypair.secretKey)
            eval_sum_keys_generated = True

        return {
            "backend": "openfhe",
            "context": context,
            "public_key": keypair.publicKey,
            "secret_key": keypair.secretKey,
            "rotate_keys_max_index": 1 if use_galois_keys else 0,
            "eval_sum_keys_generated": eval_sum_keys_generated,
        }

    raise RuntimeError(
        f"Unsupported HE backend '{backend}'. Supported backends: tenseal, openfhe"
    )


def context_from_public_bytes(ctx_bytes: bytes) -> HEContext:
    backend = get_backend_name()
    if backend == "tenseal":
        return ts.context_from(ctx_bytes)

    if backend == "openfhe":
        if not OPENFHE_AVAILABLE:
            raise RuntimeError(
                f"Unsupported HE backend '{backend}'. OpenFHE is not ready in this environment: {OPENFHE_IMPORT_ERROR}"
            )
        payload = msgpack.unpackb(ctx_bytes, raw=False)
        context = _openfhe.DeserializeCryptoContextString(
            payload["context"],
            _openfhe.BINARY,
        )
        public_key = _openfhe.DeserializePublicKeyString(
            payload["public_key"],
            _openfhe.BINARY,
        )
        return {
            "backend": "openfhe",
            "context": context,
            "public_key": public_key,
            "secret_key": None,
            "rotate_keys_max_index": 0,
            "eval_sum_keys_generated": False,
        }

    raise RuntimeError(
        f"Unsupported HE backend '{backend}'. Supported backends: tenseal, openfhe"
    )


def serialize_public_context(context: HEContext) -> bytes:
    if isinstance(context, dict) and context.get("backend") == "openfhe":
        return msgpack.packb(
            {
                "backend": "openfhe",
                "context": _openfhe.Serialize(context["context"], _openfhe.BINARY),
                "public_key": _openfhe.Serialize(context["public_key"], _openfhe.BINARY),
            },
            use_bin_type=True,
        )

    return context.serialize(
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )


def encrypt_vector(context: HEContext, values) -> HEVector:
    if isinstance(context, dict) and context.get("backend") == "openfhe":
        plaintext = context["context"].MakeCKKSPackedPlaintext(list(values))
        ciphertext = context["context"].Encrypt(context["public_key"], plaintext)
        return {
            "backend": "openfhe",
            "context": context["context"],
            "public_key": context.get("public_key"),
            "secret_key": context.get("secret_key"),
            "ciphertext": ciphertext,
            "length": len(values),
            "rotate_keys_max_index": int(context.get("rotate_keys_max_index", 0)),
            "eval_sum_keys_generated": bool(context.get("eval_sum_keys_generated", False)),
        }
    return ts.ckks_vector(context, list(values))


def decrypt_vector(vector: HEVector) -> list[float]:
    if isinstance(vector, dict) and vector.get("backend") == "openfhe":
        plaintext = vector["context"].Decrypt(vector["secret_key"], vector["ciphertext"])
        plaintext.SetLength(vector["length"])
        return [float(complex_val.real) for complex_val in plaintext.GetCKKSPackedValue()]
    return vector.decrypt()


def serialize_vector(vector: HEVector) -> bytes:
    if isinstance(vector, dict) and vector.get("backend") == "openfhe":
        return msgpack.packb(
            {
                "backend": "openfhe",
                "ciphertext": _openfhe.Serialize(vector["ciphertext"], _openfhe.BINARY),
                "length": int(vector.get("length", 0)),
            },
            use_bin_type=True,
        )
    return vector.serialize()


def vector_from_bytes(context: HEContext, raw: bytes) -> HEVector:
    if isinstance(context, dict) and context.get("backend") == "openfhe":
        try:
            payload = msgpack.unpackb(raw, raw=False)
            ciphertext_raw = payload.get("ciphertext", raw)
            length = int(payload.get("length") or (context["context"].GetRingDimension() // 2))
        except Exception:
            ciphertext_raw = raw
            length = context["context"].GetRingDimension() // 2
        ciphertext = _openfhe.DeserializeCiphertextString(ciphertext_raw, _openfhe.BINARY)
        return {
            "backend": "openfhe",
            "context": context["context"],
            "public_key": context.get("public_key"),
            "secret_key": context.get("secret_key"),
            "ciphertext": ciphertext,
            "length": length,
            "rotate_keys_max_index": int(context.get("rotate_keys_max_index", 0)),
            "eval_sum_keys_generated": bool(context.get("eval_sum_keys_generated", False)),
        }
    return ts.ckks_vector_from(context, raw)


def clone_vector(vector: HEVector) -> HEVector:
    if isinstance(vector, dict) and vector.get("backend") == "openfhe":
        ciphertext = _openfhe.DeserializeCiphertextString(
            _openfhe.Serialize(vector["ciphertext"], _openfhe.BINARY),
            _openfhe.BINARY,
        )
        cloned = dict(vector)
        cloned["ciphertext"] = ciphertext
        return cloned
    return vector_from_bytes(vector.context(), serialize_vector(vector))


def matmul(vector: HEVector, matrix) -> HEVector:
    if isinstance(vector, dict) and vector.get("backend") == "openfhe":
        context = vector["context"]
        secret_key = vector.get("secret_key")
        if secret_key is None:
            raise RuntimeError("OpenFHE matmul requires secret_key in backend context")

        if isinstance(matrix, list):
            matrix_np = np.array(matrix, dtype=np.float64)
        else:
            matrix_np = matrix

        if len(matrix_np.shape) != 2:
            raise ValueError("matrix must be 2D")

        d_in, d_out = matrix_np.shape
        _ensure_openfhe_eval_sum_keys(vector)
        _ensure_openfhe_rotate_keys(vector, max(1, d_out - 1))

        plaintext_columns = _get_openfhe_plaintext_columns(
            context=context,
            matrix_obj=matrix,
            matrix_np=matrix_np,
        )

        inner_products = []
        for col_idx in range(d_out):
            plaintext_col = plaintext_columns[col_idx]
            inner = context.EvalInnerProduct(vector["ciphertext"], plaintext_col, d_in)
            inner_products.append(inner)

        merged = context.EvalMerge(inner_products)
        result = dict(vector)
        result["ciphertext"] = merged
        result["length"] = int(d_out)
        return result
    if isinstance(matrix, list):
        return vector.mm(matrix)
    return vector.mm(matrix.tolist())


def square(vector: HEVector) -> HEVector:
    if isinstance(vector, dict) and vector.get("backend") == "openfhe":
        squared = vector["context"].EvalMult(vector["ciphertext"], vector["ciphertext"])
        result = dict(vector)
        result["ciphertext"] = squared
        return result
    return vector.square()


def _ensure_openfhe_eval_sum_keys(vector: dict[str, Any]) -> None:
    if vector.get("eval_sum_keys_generated"):
        return
    secret_key = vector.get("secret_key")
    if secret_key is None:
        raise RuntimeError("OpenFHE EvalSumKeyGen requires secret_key")
    vector["context"].EvalSumKeyGen(secret_key)
    vector["eval_sum_keys_generated"] = True


def _ensure_openfhe_rotate_keys(vector: dict[str, Any], max_index: int) -> None:
    current_max = int(vector.get("rotate_keys_max_index", 0))
    if max_index <= current_max:
        return

    secret_key = vector.get("secret_key")
    if secret_key is None:
        raise RuntimeError("OpenFHE EvalRotateKeyGen requires secret_key")

    start = max(current_max + 1, 1)
    indices = list(range(start, max_index + 1)) + list(range(-start, -max_index - 1, -1))
    vector["context"].EvalRotateKeyGen(secret_key, indices)
    vector["rotate_keys_max_index"] = max_index


def _get_openfhe_plaintext_columns(
    *,
    context,
    matrix_obj,
    matrix_np,
) -> list[Any]:
    d_in, d_out = matrix_np.shape
    cache_key = (id(context), id(matrix_obj), int(d_in), int(d_out))

    with _OPENFHE_PLAINTEXT_CACHE_LOCK:
        cached = _OPENFHE_PLAINTEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    plaintext_columns = [
        context.MakeCKKSPackedPlaintext(matrix_np[:, col_idx].tolist())
        for col_idx in range(d_out)
    ]

    with _OPENFHE_PLAINTEXT_CACHE_LOCK:
        _OPENFHE_PLAINTEXT_CACHE[cache_key] = plaintext_columns
    return plaintext_columns
