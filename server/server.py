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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup on shutdown."""
    logger.info("Server starting — loading model...")
    load_model()
    logger.info("Model loaded, server ready.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(title="ZK-LLM-Turbo Server (Milestone 4)", lifespan=lifespan)

app.include_router(inference_router)
app.include_router(session_router)

if __name__ == "__main__":
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=True)
