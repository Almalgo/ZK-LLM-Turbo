#!/usr/bin/env bash
set -euo pipefail

PROMPT="${PROMPT:-The capital of France is}"
NUM_TOKENS="${NUM_TOKENS:-5}"
NUM_ENCRYPTED_LAYERS="${NUM_ENCRYPTED_LAYERS:-1}"

python -m client.client \
  --prompt "$PROMPT" \
  --num-tokens "$NUM_TOKENS" \
  --num-encrypted-layers "$NUM_ENCRYPTED_LAYERS" \
  --stats
