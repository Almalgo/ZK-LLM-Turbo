from fastapi import FastAPI
from server.handlers.inference_handler import router
import uvicorn
from dotenv import load_dotenv
from common.logging_utils import get_logger, timed_execution
logger = get_logger("server")

# Load .env credentials
load_dotenv("server/config/credentials.env")

app = FastAPI(title="ZK-LLMS Server (Phase 2)")

# Register router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
