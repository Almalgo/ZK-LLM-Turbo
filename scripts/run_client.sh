#!/usr/bin/env bash
set -euo pipefail

PROMPT="${PROMPT:-The capital of France is}"
NUM_TOKENS="${NUM_TOKENS:-5}"
NUM_ENCRYPTED_LAYERS="${NUM_ENCRYPTED_LAYERS:-1}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[client] neither python3 nor python was found" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m client.client \
  --prompt "$PROMPT" \
  --num-tokens "$NUM_TOKENS" \
  --num-encrypted-layers "$NUM_ENCRYPTED_LAYERS" \
  --stats
