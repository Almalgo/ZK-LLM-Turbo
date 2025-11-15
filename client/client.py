import sys
from pathlib import Path
import uuid
# Ensure the project root is on the module search path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import requests, json, base64, time
import yaml
from model.tokenizer_loader import load_tokenizer, tokenize_prompt
from model.embedding_extractor import extract_embeddings
from encryption.ckks_context import create_ckks_context
from encryption.encrypt_embeddings import encrypt_embeddings
from encryption.utils import serialize_encrypted_vectors, build_payload
from pathlib import Path
from common.logging_utils import get_logger, timed_execution
logger = get_logger("client")


def load_server_config():
    cfg = yaml.safe_load(Path("client/config/endpoints.yaml").read_text())["server"]
    return cfg


def create_ckks_context():
    """Load CKKS context from config file."""
    cfg_path = Path(__file__).resolve().parents[1] / "client" / "config" / "client_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())["ckks"]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg["poly_modulus_degree"],
        coeff_mod_bit_sizes=cfg["coeff_mod_bit_sizes"],
    )
    context.global_scale = cfg["global_scale"]
    context.generate_galois_keys()
    return context



def main():
    correlation_id = str(uuid.uuid4())

    # Step 1: Tokenize
    tokenizer = load_tokenizer()
    prompt = "Explain the benefits of homomorphic encryption."
    tokens = tokenize_prompt(prompt, tokenizer)
    input_ids = tokens["input_ids"]

    # Step 2: Extract embeddings
    embeddings = extract_embeddings(input_ids)
    
    # Step 3: Create CKKS context and encrypt embeddings
    context = create_ckks_context()
    encrypted_vectors = encrypt_embeddings(embeddings, context)
    serialized = serialize_encrypted_vectors(encrypted_vectors)

    # Step 4: Build payload and send securely
    client_cfg = yaml.safe_load(Path("client/config/client_config.yaml").read_text())["ckks"]
    payload = build_payload(serialized, embeddings.shape, client_cfg)
    server_cfg = load_server_config()

    response = requests.post(
        server_cfg["base_url"] + server_cfg["infer_endpoint"],
        json=payload,
        headers={"Authorization": f"Bearer {server_cfg['auth_token']}"},
        timeout=30
    )

    print(f"Response [{response.status_code}]:", response.text)
    payload = {"encrypted_embeddings": encrypted_embeddings, "metadata": {"cid": correlation_id}}

    with timed_execution(logger, "HTTP POST /api/infer"):
        response = requests.post(
            f"{server_url}{infer_endpoint}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

    size_bytes = len(json.dumps(payload).encode("utf-8"))
    logger.info(
        "Client request sent",
        extra={"extra": {"cid": correlation_id, "payload_bytes": size_bytes, "status_code": response.status_code}}
    )

if __name__ == "__main__":
    main()
