from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import base64
import tenseal as ts
import yaml
from pathlib import Path
import json
import time
import uuid
from common.logging_utils import get_logger, timed_execution

logger = get_logger("server")
router = APIRouter()

# ---------------------------
# Data Model
# ---------------------------
class EmbeddingRequest(BaseModel):
    encrypted_embeddings: list[str]
    metadata: dict

# ---------------------------
# Utility
# ---------------------------
def load_ckks_context():
    """Load CKKS encryption context from config file."""
    config_path = Path(__file__).resolve().parents[1] / "config" / "server_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"[Server] Missing CKKS config: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())["ckks"]

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=cfg["poly_modulus_degree"],
        coeff_mod_bit_sizes=cfg["coeff_mod_bit_sizes"],
    )
    context.global_scale = cfg["global_scale"]
    context.generate_galois_keys()
    return context

# ---------------------------
# Inference Endpoint
# ---------------------------
@router.post("/api/infer")
async def infer(request: Request):
    """Receive encrypted embeddings, decrypt for testing, and return dummy encrypted result."""
    start_time = time.perf_counter()
    cid = None

    try:
        # --------------------
        # Parse incoming JSON
        # --------------------
        body = await request.json()
        cid = body.get("metadata", {}).get("cid", str(uuid.uuid4()))
        size_bytes = len(json.dumps(body).encode("utf-8"))
        logger.info("Received request", extra={"extra": {"cid": cid, "payload_bytes": size_bytes}})
    except Exception as e:
        logger.error("Failed to parse request", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=400, detail="Invalid request format")

    try:
        # --------------------
        # Load encryption context
        # --------------------
        context = load_ckks_context()

        # --------------------
        # Decrypt embeddings (for testing)
        # --------------------
        encrypted_embeddings = body.get("encrypted_embeddings", [])
        metadata = body.get("metadata", {})

        with timed_execution(logger, f"Decrypt embeddings for {cid}"):
            decrypted_embeddings = []
            for enc_b64 in encrypted_embeddings:
                enc_bytes = base64.b64decode(enc_b64)
                enc_vec = ts.ckks_vector_from(context, enc_bytes)
                decrypted_embeddings.append(enc_vec.decrypt())

        logger.info(
            "Decryption complete",
            extra={"extra": {"cid": cid, "embeddings_count": len(decrypted_embeddings)}},
        )

        if decrypted_embeddings:
            logger.info(
                "Example decrypted slice",
                extra={"extra": {"cid": cid, "values": decrypted_embeddings[0][:5]}},
            )

        # --------------------
        # Dummy encrypted computation
        # --------------------
        dummy_result = [0.1, 0.2, 0.3]
        enc_result = ts.ckks_vector(context, dummy_result)
        result_b64 = base64.b64encode(enc_result.serialize()).decode("utf-8")

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Request completed", extra={"extra": {"cid": cid, "elapsed_ms": round(elapsed_ms, 2)}})

        return {"encrypted_result": result_b64}

    except FileNotFoundError as e:
        logger.error("Missing configuration file", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error("Server error", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=500, detail="Internal server error")
