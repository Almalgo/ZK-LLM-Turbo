import tenseal as ts
import yaml
from pathlib import Path


def create_ckks_context(config_path: str = "client/config/client_config.yaml"):
    """Create and return a TenSEAL CKKS context from config file."""
    cfg = yaml.safe_load(Path(config_path).read_text())["ckks"]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg["poly_modulus_degree"],
        coeff_mod_bit_sizes=cfg["coeff_mod_bit_sizes"],
    )
    context.global_scale = cfg["global_scale"]

    if cfg.get("use_galois_keys", False):
        context.generate_galois_keys()
    if cfg.get("use_relin_keys", False):
        context.generate_relin_keys()

    return context


def serialize_public_context(context: ts.Context) -> bytes:
    """Serialize context without secret key (public context only).

    The resulting bytes contain the public key, galois keys, and relin keys,
    but NOT the secret key. The server can use this to perform HE operations
    but cannot decrypt any ciphertexts.
    """
    return context.serialize(
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )
