import asyncio
import importlib.util
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from server.handlers.inference_handler import router as inference_router
from server.handlers.session_handler import (
    cleanup_expired_sessions,
    load_session_config,
    router as session_router,
)
from server.model.weight_manager import load_model
import uvicorn
from dotenv import load_dotenv
from common.logging_utils import get_logger

logger = get_logger("server")

load_dotenv("server/config/credentials.env")


def _check_hexl():
    """Log whether Intel HEXL acceleration is actually available to TenSEAL."""
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read().lower()
        avx512_detected = "avx512" in cpuinfo
    except Exception:
        avx512_detected = False

    tenseal_binaries = _find_tenseal_binaries()
    linked_binaries = [str(path) for path in tenseal_binaries if _binary_mentions_hexl(path)]

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
                "probed_binaries": [str(path) for path in tenseal_binaries],
            }},
        )
    else:
        logger.warning(
            "AVX512 not detected — Intel HEXL acceleration unavailable. See docs/intel-hexl-build.md",
            extra={"extra": {"probed_binaries": [str(path) for path in tenseal_binaries]}},
        )


def _find_tenseal_binaries() -> list[Path]:
    """Find candidate native TenSEAL/SEAL binaries for linkage inspection."""
    paths = []
    spec = importlib.util.find_spec("tenseal")
    if spec and spec.origin:
        pkg_dir = Path(spec.origin).resolve().parent
        site_dir = pkg_dir.parent
        candidates = [
            site_dir / "libtenseal.so",
            site_dir / "_sealapi_cpp.cpython-313-x86_64-linux-gnu.so",
            site_dir / "_tenseal_cpp.cpython-313-x86_64-linux-gnu.so",
        ]
        paths.extend(path for path in candidates if path.exists())
        paths.extend(sorted(site_dir.glob("_sealapi_cpp*.so")))
        paths.extend(sorted(site_dir.glob("_tenseal_cpp*.so")))
    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _binary_mentions_hexl(binary_path: Path) -> bool:
    """Probe a native binary for signs that Intel HEXL is linked or embedded."""
    commands = [
        ["ldd", str(binary_path)],
        ["strings", str(binary_path)],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        haystack = (proc.stdout + "\n" + proc.stderr).lower()
        if "hexl" in haystack or "seal_use_intel_hexl" in haystack:
            return True
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup on shutdown."""
    logger.info("Server starting — loading model...")
    load_model()
    _check_hexl()
    session_cfg = load_session_config()

    async def _session_cleanup_loop():
        while True:
            cleanup_expired_sessions(session_cfg["session_ttl_seconds"])
            await asyncio.sleep(session_cfg["cleanup_interval_seconds"])

    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    logger.info("Model loaded, server ready.")
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

app.include_router(inference_router)
app.include_router(session_router)

if __name__ == "__main__":
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=True)
