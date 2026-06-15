from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from server.handlers.inference_handler import router as inference_router
from server.handlers.session_handler import router as session_router
from server.model.weight_manager import MODEL_NAME, load_model
import uvicorn
from dotenv import load_dotenv
from common.logging_utils import get_logger

logger = get_logger("server")

load_dotenv("server/config/credentials.env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup on shutdown."""
    app.state.model_ready = False
    app.state.model_name = MODEL_NAME
    logger.info("Server starting — loading model...")
    try:
        load_model()
        app.state.model_ready = True
        logger.info("Model loaded, server ready.")
        yield
    finally:
        app.state.model_ready = False
        logger.info("Server shutting down.")


app = FastAPI(title="ZK-LLM-Turbo Server (Milestone 4)", lifespan=lifespan)
app.state.model_ready = False
app.state.model_name = MODEL_NAME


@app.get("/health")
async def health():
    """Lightweight liveness check that does not touch model state."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    """Readiness check used by Coolify and HAAS deployment diagnostics."""
    if getattr(request.app.state, "model_ready", False):
        return {
            "status": "ready",
            "model": getattr(request.app.state, "model_name", MODEL_NAME),
        }
    return JSONResponse(status_code=503, content={"status": "starting"})

app.include_router(inference_router)
app.include_router(session_router)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.server:app", host=host, port=port, reload=True)
