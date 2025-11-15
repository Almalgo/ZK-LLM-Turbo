import json
import logging
import sys
import time
import uuid

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_record.update(record.extra)
        return json.dumps(log_record)

def get_logger(name: str):
    """Return a logger with JSON formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def timed_execution(logger, label):
    """Context manager to time code blocks."""
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = (time.perf_counter() - self.start) * 1000
            logger.info(f"{label} completed", extra={"extra": {"elapsed_ms": round(elapsed, 2)}})
    return _Timer()
