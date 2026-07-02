import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from server.handlers.inference_handler import router as inference_router
from server.handlers.session_handler import (
    cleanup_expired_sessions,
    load_session_config,
    router as session_router,
)
from server.model.weight_manager import get_model_status
import uvicorn
from dotenv import load_dotenv
from common.logging_utils import get_logger
from common.he_backend import get_backend_status
from common.hexl_probe import probe_hexl_linkage

logger = get_logger("server")

load_dotenv("server/config/credentials.env")

SERVICE_NAME = "zk-llm-turbo"
SNET_SERVICE_ID = "zk_llm1"


def _check_hexl():
    """Log whether Intel HEXL acceleration is actually available to TenSEAL."""
    probe = probe_hexl_linkage()
    avx512_detected = probe["avx512_detected"]
    linked_binaries = probe["linked_binaries"]
    probed_binaries = probe["probed_binaries"]

    if linked_binaries:
        logger.info(
            "Intel HEXL linked",
            extra={"extra": {
                "avx512_detected": avx512_detected,
                "linked_binaries": linked_binaries,
            }},
        )
        return

    if avx512_detected:
        logger.warning(
            "AVX512 detected but current TenSEAL build does not appear HEXL-linked. See docs/intel-hexl-build.md",
            extra={"extra": {
                "avx512_detected": avx512_detected,
                "probed_binaries": probed_binaries,
            }},
        )
    else:
        logger.warning(
            "AVX512 not detected — Intel HEXL acceleration unavailable. See docs/intel-hexl-build.md",
            extra={"extra": {"probed_binaries": probed_binaries}},
        )


def _check_he_backend():
    status = get_backend_status()
    selected_backend = status["selected_backend"]

    if selected_backend == "openfhe":
        logger.warning(
            "OpenFHE backend selected. Matmul path is experimental and not yet benchmark-validated for production.",
            extra={"extra": status},
        )
        return

    logger.info(
        "HE backend status",
        extra={"extra": status},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start lightweight process services without blocking health checks."""
    logger.info("Server starting")
    _check_he_backend()
    _check_hexl()
    session_cfg = load_session_config()

    async def _session_cleanup_loop():
        while True:
            cleanup_expired_sessions(session_cfg["session_ttl_seconds"])
            await asyncio.sleep(session_cfg["cleanup_interval_seconds"])

    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    logger.info("Server ready")
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Server shutting down.")


app = FastAPI(title="ZK-LLM-Turbo Server (Milestone 4)", lifespan=lifespan)


def _liveness_payload() -> dict:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "serviceID": SNET_SERVICE_ID,
    }


def _health_payload() -> dict:
    """Return lightweight health without triggering lazy model loading."""
    return {
        **_liveness_payload(),
        **get_model_status(),
    }


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/health")
async def health() -> dict:
    return _health_payload()


@app.post("/health")
async def health_rpc() -> dict:
    """SNET Publisher HTTP proto maps Health RPC to POST /health."""
    return _health_payload()


@app.get("/heartbeat")
async def heartbeat() -> dict:
    return _liveness_payload()


@app.post("/heartbeat")
async def heartbeat_rpc() -> dict:
    """Accept POST heartbeats for daemon/proxy implementations that use RPC style."""
    return _liveness_payload()

app.include_router(inference_router)
app.include_router(session_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.server:app", host="0.0.0.0", port=port, reload=True)
