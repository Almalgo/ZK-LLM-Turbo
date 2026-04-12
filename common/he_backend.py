"""Backend abstraction for HE context and vector operations."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import msgpack
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

        keypair = context.KeyGen()
        if use_relin_keys:
            context.EvalMultKeyGen(keypair.secretKey)
        if use_galois_keys:
            context.EvalRotateKeyGen(keypair.secretKey, [1, -1])

        return {
            "backend": "openfhe",
            "context": context,
            "public_key": keypair.publicKey,
            "secret_key": keypair.secretKey,
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
        raise NotImplementedError("OpenFHE backend matmul is not implemented yet")
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
