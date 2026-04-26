"""Server security utilities for request authentication."""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status
from typing import Annotated


def _get_required_token() -> str:
    return (
        os.getenv("ZKLLM_SERVER_AUTH_TOKEN")
        or os.getenv("ZKLLM_API_TOKEN")
        or os.getenv("AUTH_TOKEN")
        or ""
    ).strip()


def _auth_required() -> bool:
    # Auth is enabled unless explicitly disabled.
    # Set ZKLLM_REQUIRE_API_TOKEN=false to disable.
    env_flag = os.getenv("ZKLLM_REQUIRE_API_TOKEN", "").strip().lower()
    if env_flag in {"0", "false", "no", "off"}:
        return False
    return True


def _is_secure_env_enabled() -> bool:
    # Security can be explicitly disabled with
    # ZKLLM_REQUIRE_API_TOKEN=false
    return _auth_required()


def validate_bearer_token(
    authorization: Annotated[str | None, Header(default=None)] = None,
):
    if not _is_secure_env_enabled():
        return

    expected = _get_required_token()
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed Authorization header",
        )

    provided = authorization[len("Bearer "):]
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication token",
        )
