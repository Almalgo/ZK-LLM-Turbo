import runpod
import io
import sys
import importlib
import os
import json
import sentry_sdk
from datetime import datetime, timezone

SENTRY_DSN = os.getenv("SENTRY_DSN")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

sentry_sdk.init(
    dsn=SENTRY_DSN,
    traces_sample_rate=1.0,
    send_default_pii=False,
)


def log_to_sentry(message: str, level: str = "info", source: str = "handler", **kwargs) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    if kwargs:
        data_str = json.dumps(kwargs, separators=(',', ':'))
        formatted_message = f"{timestamp} [{source}] [{level.upper()}] {message} | {data_str}"
    else:
        formatted_message = f"{timestamp} [{source}] [{level.upper()}] {message}"

    with sentry_sdk.push_scope() as scope:
        scope.set_tag("endpoint_id", ENDPOINT_ID)
        scope.set_tag("source", source)
        scope.set_tag("log_level", level)

        sentry_sdk.capture_message(formatted_message, level=level)


def handler(event):
    log_buffer = io.StringIO()
    original_stdout = sys.stdout

    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("endpoint_id", ENDPOINT_ID)

    try:
        log_to_sentry(
            "Handler started",
            level="info",
            input_data=event.get("input", {}),
        )

        sys.stdout = log_buffer
        user_module = importlib.import_module("customer_main")
        result = user_module.run(event["input"])
        sys.stdout = original_stdout

        customer_logs = log_buffer.getvalue()

        if customer_logs.strip():
            for line in customer_logs.strip().split('\n'):
                if line.strip():
                    log_to_sentry(
                        line.strip(),
                        level="info",
                        source="customer_code"
                    )

        log_to_sentry(
            "Run completed successfully",
            level="info",
            result_type=type(result).__name__,
            result_preview=str(result)[:200],
        )

        return {"result": result}

    except Exception as e:
        sys.stdout = original_stdout
        customer_logs = log_buffer.getvalue()

        if customer_logs.strip():
            log_to_sentry(
                "Customer logs before error",
                level="warning",
                source="customer_code",
                logs_preview=customer_logs[:500],
            )

        log_to_sentry(
            f"Handler failed: {str(e)}",
            level="error",
            source="error",
            error_type=type(e).__name__,
        )

        sentry_sdk.capture_exception(e)

        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "logs": customer_logs
        }

    finally:
        sys.stdout = original_stdout


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
