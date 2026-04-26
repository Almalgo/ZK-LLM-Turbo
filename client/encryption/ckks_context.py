import yaml
from pathlib import Path

from common.he_backend import create_context as create_backend_context
from common.he_backend import serialize_public_context as serialize_backend_public_context


def load_ckks_config(config_path: str = "client/config/client_config.yaml") -> dict:
    """Load the CKKS section from the client config file."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        payload = yaml.load(f, Loader=yaml.FullLoader)
    return payload["ckks"]


def create_ckks_context(
    config_path: str = "client/config/client_config.yaml",
    global_scale_override: int | None = None,
    cfg: dict | None = None,
):
    """Create and return a TenSEAL CKKS context from config file."""
    config = cfg if cfg is not None else load_ckks_config(config_path)

    return create_backend_context(
        poly_modulus_degree=config["poly_modulus_degree"],
        coeff_mod_bit_sizes=config["coeff_mod_bit_sizes"],
        global_scale=(
            global_scale_override if global_scale_override is not None else config["global_scale"]
        ),
        use_galois_keys=config.get("use_galois_keys", False),
        use_relin_keys=config.get("use_relin_keys", False),
    )


def serialize_public_context(context) -> bytes:
    """Serialize context without secret key (public context only).

    The resulting bytes contain the public key, galois keys, and relin keys,
    but NOT the secret key. The server can use this to perform HE operations
    but cannot decrypt any ciphertexts.
    """
    return serialize_backend_public_context(context)
