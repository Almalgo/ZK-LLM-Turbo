from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import msgpack
import uuid
from common.logging_utils import get_logger
from server.model.weight_manager import ModelUnavailableError
from server.services.layer_service import process_binary_payload, process_layer_request

logger = get_logger("server.inference")
router = APIRouter()


class LayerRequest(BaseModel):
    session_id: str
    layer_idx: int
    operation: str  # "qkv", "o_proj", "ffn_gate_up", "ffn_down", "ffn_merged"
    encrypted_vectors_b64: list[str]
    chunk_sizes: list[int] | None = None  # for split-input operations
    pack_counts: list[int] | None = None  # informational only


class LayerResponse(BaseModel):
    encrypted_results_b64: list[str]
    operation: str
    layer_idx: int
    elapsed_ms: float


def _process_binary_payload(req_data: dict, cid: str) -> bytes:
    return process_binary_payload(req_data, cid)


@router.post("/api/layer", response_model=LayerResponse)
async def process_layer(req: LayerRequest):
    """Process one operation of the split-inference protocol.

    The server uses only the PUBLIC context (no secret key)
    to perform linear algebra on encrypted data.
    """
    cid = str(uuid.uuid4())

    try:
        return LayerResponse(**process_layer_request(req.model_dump(), cid=cid))

    except HTTPException:
        raise
    except ModelUnavailableError as e:
        logger.error("Model unavailable", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=503, detail=f"Model is not available: {e}")
    except Exception as e:
        logger.error("Layer op failed", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@router.post("/api/layer/binary")
async def process_layer_binary(request: Request):
    """Binary msgpack endpoint — no base64, no JSON overhead."""
    cid = str(uuid.uuid4())

    try:
        body = await request.body()

        # Defense-in-depth: reject oversized compressed payloads before decompression
        MAX_COMPRESSED_PAYLOAD = 10_000_000  # 10 MB
        if len(body) > MAX_COMPRESSED_PAYLOAD:
            raise HTTPException(status_code=413, detail="Compressed payload too large")

        req_data = msgpack.unpackb(body, raw=False)

        response_data = _process_binary_payload(req_data, cid)
        return Response(content=response_data, media_type="application/msgpack")

    except HTTPException:
        raise
    except ModelUnavailableError as e:
        logger.error("Model unavailable (binary)", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=503, detail=f"Model is not available: {e}")
    except Exception as e:
        logger.error("Layer op failed (binary)", extra={"extra": {"cid": cid, "error": str(e)}})
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@router.websocket("/api/layer/ws")
async def process_layer_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            body = await websocket.receive_bytes()
            if len(body) > 10_000_000:
                raise HTTPException(status_code=413, detail="Compressed payload too large")

            req_data = msgpack.unpackb(body, raw=False)
            cid = str(uuid.uuid4())
            response_data = _process_binary_payload(req_data, cid)
            await websocket.send_bytes(response_data)
    except WebSocketDisconnect:
        logger.info("Layer websocket disconnected")
    except HTTPException as exc:
        await websocket.close(code=1008, reason=exc.detail)
    except ModelUnavailableError as e:
        logger.error("Model unavailable (websocket)", extra={"extra": {"error": str(e)}})
        await websocket.close(code=1013, reason="Model is not available")
    except Exception as e:
        logger.error("Layer websocket failed", extra={"extra": {"error": str(e)}})
        await websocket.close(code=1011, reason="Inference error")


# Keep the old /api/infer endpoint for backward compatibility
class EmbeddingRequest(BaseModel):
    encrypted_embeddings: list[str]
    metadata: dict


@router.post("/api/infer")
async def infer(req: EmbeddingRequest):
    """Legacy endpoint — returns dummy result for backward compatibility."""
    return {"encrypted_result": "", "message": "Use /api/session + /api/layer for real inference"}
