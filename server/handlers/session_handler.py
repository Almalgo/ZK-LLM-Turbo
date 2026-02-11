import base64
import uuid
import tenseal as ts
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from common.logging_utils import get_logger

logger = get_logger("server.session")
router = APIRouter()

# In-memory session store: session_id -> TenSEAL public context
_sessions: dict[str, ts.Context] = {}


class SessionRequest(BaseModel):
    public_context_b64: str


class SessionResponse(BaseModel):
    session_id: str


@router.post("/api/session", response_model=SessionResponse)
async def create_session(req: SessionRequest):
    """Receive a public CKKS context (no secret key) and create a session."""
    try:
        ctx_bytes = base64.b64decode(req.public_context_b64)
        context = ts.context_from(ctx_bytes)
        session_id = str(uuid.uuid4())
        _sessions[session_id] = context
        logger.info("Session created", extra={"extra": {"session_id": session_id}})
        return SessionResponse(session_id=session_id)
    except Exception as e:
        logger.error("Failed to create session", extra={"extra": {"error": str(e)}})
        raise HTTPException(status_code=400, detail=f"Invalid public context: {e}")


def get_session(session_id: str) -> ts.Context:
    """Retrieve the public context for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return _sessions[session_id]
