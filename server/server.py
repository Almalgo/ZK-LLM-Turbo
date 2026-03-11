from contextlib import asynccontextmanager
from fastapi import FastAPI
from server.handlers.inference_handler import router as inference_router
from server.handlers.session_handler import router as session_router
from server.model.weight_manager import load_model
import uvicorn
from dotenv import load_dotenv
from common.logging_utils import get_logger

logger = get_logger("server")

load_dotenv("server/config/credentials.env")


def _check_hexl():
    """Log whether Intel HEXL acceleration is likely available."""
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "avx512" in cpuinfo.lower():
            logger.info("AVX512 detected — Intel HEXL acceleration may be active if SEAL was compiled with HEXL support.")
        else:
            logger.warning("AVX512 not detected — Intel HEXL acceleration unavailable. See docs/intel-hexl-integration.md")
    except Exception:
        logger.warning("Could not check CPU features for HEXL support.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup on shutdown."""
    logger.info("Server starting — loading model...")
    load_model()
    _check_hexl()
    logger.info("Model loaded, server ready.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="ZK-LLM-Turbo Server (Milestone 4)", lifespan=lifespan)

app.include_router(inference_router)
app.include_router(session_router)

if __name__ == "__main__":
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=True)
