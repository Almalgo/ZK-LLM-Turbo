#!/usr/bin/env sh
set -eu

# One image can support both deployment targets:
# - SNET Full-Stack HaaS/RunPod: default when no HTTP port is provided.
# - Coolify/FastAPI: selected explicitly with SERVICE_MODE=fastapi or implicitly
#   when Coolify injects PORT.
MODE="${SERVICE_MODE:-}"
if [ -z "$MODE" ]; then
  if [ -n "${PORT:-}" ]; then
    MODE="fastapi"
  else
    MODE="runpod"
  fi
fi

case "$MODE" in
  fastapi|http|coolify)
    exec uvicorn server.server:app --host 0.0.0.0 --port "${PORT:-8000}"
    ;;
  runpod|haas|snet)
    exec python -u runpod_handler.py
    ;;
  *)
    echo "Unsupported SERVICE_MODE=$MODE. Use fastapi or runpod." >&2
    exit 64
    ;;
esac
