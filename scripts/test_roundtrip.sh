#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PROMPT="${PROMPT:-The capital of France is}"
NUM_TOKENS="${NUM_TOKENS:-5}"
NUM_ENCRYPTED_LAYERS="${NUM_ENCRYPTED_LAYERS:-1}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-300}"

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[roundtrip] Starting server on ${HOST}:${PORT}..."
HOST="$HOST" PORT="$PORT" bash scripts/run_server.sh > /tmp/zk_llm_server.log 2>&1 &
SERVER_PID=$!

echo "[roundtrip] Waiting for server to become ready..."
START_TS=$(date +%s)
until curl -fsS "http://${HOST}:${PORT}/docs" >/dev/null 2>&1; do
  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))
  if [[ "$ELAPSED" -ge "$STARTUP_TIMEOUT_SEC" ]]; then
    echo "[roundtrip] Server did not become ready within ${STARTUP_TIMEOUT_SEC}s"
    echo "[roundtrip] --- server log tail ---"
    tail -n 80 /tmp/zk_llm_server.log || true
    exit 1
  fi
  sleep 2
done

echo "[roundtrip] Server is ready. Running client..."
python -m client.client \
  --prompt "$PROMPT" \
  --num-tokens "$NUM_TOKENS" \
  --num-encrypted-layers "$NUM_ENCRYPTED_LAYERS" \
  --stats

echo "[roundtrip] Completed successfully."
