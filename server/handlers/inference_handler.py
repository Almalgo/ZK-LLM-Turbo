from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import tenseal as ts
import time
import uuid
from common.logging_utils import get_logger
from server.handlers.session_handler import get_session
from server.model.weight_manager import get_layer_weights
from server.inference.he_ops import (
    compute_qkv_projections,
    compute_o_projection,
    compute_ffn_gate_up,
    compute_ffn_down,
)

logger = get_logger("server.inference")
router = APIRouter()


class LayerRequest(BaseModel):
    session_id: str
    layer_idx: int
    operation: str  # "qkv", "o_proj", "ffn_gate_up", "ffn_down"
    encrypted_vectors_b64: list[str]
    chunk_sizes: list[int] | None = None  # for split-input operations


class LayerResponse(BaseModel):
    encrypted_results_b64: list[str]
    operation: str
    layer_idx: int
    elapsed_ms: float


def _deserialize_vectors(
    context: ts.Context, vectors_b64: list[str]
) -> list[ts.CKKSVector]:
    """Deserialize base64-encoded encrypted vectors using the session's public context."""
    result = []
    for b64 in vectors_b64:
        raw = base64.b64decode(b64)
        vec = ts.ckks_vector_from(context, raw)
        result.append(vec)
    return result


def _serialize_vectors(vectors: list[ts.CKKSVector]) -> list[str]:
    """Serialize encrypted vectors to base64."""
    return [base64.b64encode(v.serialize()).decode("utf-8") for v in vectors]


@router.post("/api/layer", response_model=LayerResponse)
async def process_layer(req: LayerRequest):
    """Process one operation of the split-inference protocol.

    The server uses only the PUBLIC context (no secret key)
    to perform linear algebra on encrypted data.
    """
    start = time.perf_counter()
    cid = str(uuid.uuid4())

    try:
        context = get_session(req.session_id)
        weights = get_layer_weights(req.layer_idx)
        enc_vectors = _deserialize_vectors(context, req.encrypted_vectors_b64)

        logger.info(
            "Processing layer op",
            extra={"extra": {
                "cid": cid, "layer": req.layer_idx,
                "op": req.operation, "num_vectors": len(enc_vectors),
            }},
        )

        if req.operation == "qkv":
            # Input: 1 encrypted normed hidden state
            qkv = compute_qkv_projections(enc_vectors[0], weights)
            result_vectors = [qkv["q"], qkv["k"], qkv["v"]]

        elif req.operation == "o_proj":
            # Input: 1 encrypted attention output
            o_out = compute_o_projection(enc_vectors[0], weights)
            result_vectors = [o_out]

        elif req.operation == "ffn_gate_up":
            # Input: 1 encrypted normed hidden state
            gu = compute_ffn_gate_up(enc_vectors[0], weights)
            # Return gate parts then up parts
            result_vectors = gu["gate_parts"] + gu["up_parts"]

        elif req.operation == "ffn_down":
            # Input: N encrypted chunks of SiLU(gate)*up
            if req.chunk_sizes is None:
                raise HTTPException(
                    status_code=400,
                    detail="ffn_down requires chunk_sizes",
                )
            down = compute_ffn_down(enc_vectors, weights, req.chunk_sizes)
            result_vectors = [down]

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown operation: {req.operation}",
            )

        results_b64 = _serialize_vectors(result_vectors)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Layer op complete",
            extra={"extra": {
                "cid": cid, "op": req.operation,
                "elapsed_ms": round(elapsed_ms, 2),
            }},
        )

        return LayerResponse(
            encrypted_results_b64=results_b64,
            operation=req.operation,
            layer_idx=req.layer_idx,
            elapsed_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Layer op failed", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


# Keep the old /api/infer endpoint for backward compatibility
class EmbeddingRequest(BaseModel):
    encrypted_embeddings: list[str]
    metadata: dict


@router.post("/api/infer")
async def infer(req: EmbeddingRequest):
    """Legacy endpoint — returns dummy result for backward compatibility."""
    return {"encrypted_result": "", "message": "Use /api/session + /api/layer for real inference"}
