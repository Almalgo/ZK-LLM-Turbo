#!/usr/bin/env bash
set -euo pipefail

# Build and start server in background.
docker compose up -d server

echo "Waiting for server to be ready..."
until curl -fsS http://127.0.0.1:8000/docs >/dev/null 2>&1; do
  sleep 2
done

echo "Running demo client container..."
docker compose --profile demo run --rm demo-client
