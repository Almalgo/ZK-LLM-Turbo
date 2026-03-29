import base64
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import tenseal as ts
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from common.logging_utils import get_logger

logger = get_logger("server.session")
router = APIRouter()


@dataclass
class SessionEntry:
    context: ts.Context
    created_at: float
    last_accessed: float


# In-memory session store: session_id -> session entry
_sessions: dict[str, SessionEntry] = {}

_CONFIG_PATH = Path("server/config/server_config.yaml")


def load_session_config(config_path: Path = _CONFIG_PATH) -> dict:
    """Load session-management settings from server config."""
    cfg = yaml.safe_load(config_path.read_text())
    session_cfg = cfg.get("session", {})
    return {
        "max_sessions": int(session_cfg.get("max_sessions", 50)),
        "session_ttl_seconds": int(session_cfg.get("session_ttl_seconds", 3600)),
        "cleanup_interval_seconds": int(session_cfg.get("cleanup_interval_seconds", 60)),
    }


class SessionRequest(BaseModel):
    public_context_b64: str


class SessionResponse(BaseModel):
    session_id: str


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool


def cleanup_expired_sessions(max_age_seconds: int, now: float | None = None) -> int:
    """Remove sessions idle longer than the TTL."""
    now = time.time() if now is None else now
    expired_ids = [
        session_id
        for session_id, entry in _sessions.items()
        if now - entry.last_accessed > max_age_seconds
    ]
    for session_id in expired_ids:
        del _sessions[session_id]

    if expired_ids:
        logger.info("Expired sessions cleaned", extra={"extra": {
            "expired_count": len(expired_ids),
            "remaining_sessions": len(_sessions),
        }})
    return len(expired_ids)


def cleanup_excess_sessions(max_sessions: int) -> int:
    """Evict least-recently-used sessions above the configured cap."""
    if max_sessions <= 0:
        max_sessions = 1

    excess = max(0, len(_sessions) - max_sessions)
    if excess == 0:
        return 0

    evicted_ids = [
        session_id
        for session_id, _ in sorted(
            _sessions.items(),
            key=lambda item: item[1].last_accessed,
        )[:excess]
    ]
    for session_id in evicted_ids:
        del _sessions[session_id]

    logger.warning("Excess sessions evicted", extra={"extra": {
        "evicted_count": len(evicted_ids),
        "remaining_sessions": len(_sessions),
    }})
    return len(evicted_ids)


@router.post("/api/session", response_model=SessionResponse)
async def create_session(req: SessionRequest):
    """Receive a public CKKS context (no secret key) and create a session."""
    try:
        config = load_session_config()
        cleanup_expired_sessions(config["session_ttl_seconds"])

        ctx_bytes = base64.b64decode(req.public_context_b64)
        context = ts.context_from(ctx_bytes)
        session_id = str(uuid.uuid4())
        now = time.time()
        _sessions[session_id] = SessionEntry(
            context=context,
            created_at=now,
            last_accessed=now,
        )
        cleanup_excess_sessions(config["max_sessions"])
        logger.info("Session created", extra={"extra": {"session_id": session_id}})
        return SessionResponse(session_id=session_id)
    except Exception as e:
        logger.error("Failed to create session", extra={"extra": {"error": str(e)}})
        raise HTTPException(status_code=400, detail=f"Invalid public context: {e}")


def get_session(session_id: str) -> ts.Context:
    """Retrieve the public context for a session."""
    entry = _sessions.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    entry.last_accessed = time.time()
    return entry.context


@router.delete("/api/session/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """Delete a session explicitly."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    del _sessions[session_id]
    logger.info("Session deleted", extra={"extra": {"session_id": session_id}})
    return DeleteSessionResponse(session_id=session_id, deleted=True)
