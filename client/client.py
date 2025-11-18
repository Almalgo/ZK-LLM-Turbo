import sys
from pathlib import Path
import uuid
import requests, json, base64, time, yaml
from client.model.tokenizer_loader import load_tokenizer, tokenize_prompt
from client.model.embedding_extractor import extract_embeddings
from client.encryption.ckks_context import create_ckks_context
from client.encryption.encrypt_embeddings import encrypt_embeddings
from client.encryption.utils import serialize_encrypted_vectors, build_payload
import tenseal as ts


# Ensure the project root is on the module search path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from common.logging_utils import get_logger, timed_execution



logger = get_logger("client")


def load_server_config():
    cfg = yaml.safe_load(Path("client/config/endpoints.yaml").read_text())["server"]
    return cfg


def main(prompt: str = "Explain the benefits of homomorphic encryption."):
    correlation_id = str(uuid.uuid4())
    logger.info("Starting client request", extra={"extra": {"cid": correlation_id}})

    # Step 1: Tokenize
    tokenizer = load_tokenizer()
    prompt = "Explain the benefits of homomorphic encryption."
    tokens = tokenize_prompt(prompt, tokenizer)
    input_ids = tokens["input_ids"]

    # Step 2: Extract embeddings
    embeddings = extract_embeddings(input_ids)
    logger.info("Extracted embeddings", extra={"extra": {"cid": correlation_id, "shape": list(embeddings.shape)}})

    # Step 3: Create CKKS context and encrypt embeddings
    context = create_ckks_context()

    with timed_execution(logger, "Encrypt embeddings"):
        encrypted_vectors = encrypt_embeddings(embeddings, context)
        serialized = serialize_encrypted_vectors(encrypted_vectors)

    # Step 4: Build payload
    client_cfg = yaml.safe_load(Path("client/config/client_config.yaml").read_text())["ckks"]
    payload = build_payload(serialized, embeddings.shape, client_cfg)
    payload["metadata"]["cid"] = correlation_id  # attach correlation id for tracing

    size_bytes = len(json.dumps(payload).encode("utf-8"))
    logger.info("Payload prepared", extra={"extra": {"cid": correlation_id, "payload_bytes": size_bytes}})

    # Step 5: Load server configuration
    server_cfg = load_server_config()
    url = server_cfg["base_url"] + server_cfg["infer_endpoint"]

    logger.info(
    "TinyLlama embeddings ready for encryption",
    extra={"extra": {"cid": correlation_id, "seq_len": embeddings.shape[0], "dim": embeddings.shape[1]}}
)

    # Step 6: Send to server
    with timed_execution(logger, "HTTP POST /api/infer"):
        response = requests.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {server_cfg['auth_token']}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

    logger.info(
        "Response received",
        extra={"extra": {"cid": correlation_id, "status_code": response.status_code}}
    )

    if response.status_code != 200:
        logger.error(
            "Request failed",
            extra={"extra": {"cid": correlation_id, "body": response.text}},
        )
        print(f"[ERROR] {response.status_code}: {response.text}")
        return

    # Step 7: Handle response
    resp_body = response.json()
    enc_result_b64 = resp_body["encrypted_result"]
    enc_bytes = base64.b64decode(enc_result_b64)
    vec = ts.ckks_vector_from(context, enc_bytes)
    decrypted = vec.decrypt()

    logger.info(
        "Decrypted server result",
        extra={"extra": {"cid": correlation_id, "values": decrypted}},
    )
    print(f"[OK] Decrypted result: {decrypted}")


def benchmark_prompts(prompts):
    import numpy as np
    times = []
    for prompt in prompts:
        start = time.perf_counter()
        main(prompt=prompt)
        times.append((time.perf_counter() - start) * 1000)
    print(f"Average round-trip latency: {np.mean(times):.2f} ms")

if __name__ == "__main__":
    prompts = [
        "Explain the benefits of homomorphic encryption.",
        "What is secure multiparty computation?",
        "How does zero-knowledge proof work?"
    ]
    benchmark_prompts(prompts)
