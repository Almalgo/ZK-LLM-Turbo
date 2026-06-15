#!/usr/bin/env bash
set -u -o pipefail

BASE_URL="${BASE_URL:-https://zkllm.almalgo.com}"
PROMPT="${PROMPT:-The capital of France is}"
NUM_TOKENS="${NUM_TOKENS:-2}"
NUM_ENCRYPTED_LAYERS="${NUM_ENCRYPTED_LAYERS:-1}"
RUN_CLIENT="${RUN_CLIENT:-0}"
CHECK_SESSION="${CHECK_SESSION:-1}"
INSECURE="${INSECURE:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-60}"
PYTHON_BIN="${PYTHON_BIN:-}"

BASE_URL="${BASE_URL%/}"

CURL_TLS_ARGS=()
if [[ "$INSECURE" == "1" ]]; then
  CURL_TLS_ARGS=(-k)
fi

FAILED=0
BASIC_CHECKS_PASSED=1
CORE_API_PASSED=1
SESSION_CHECK_PASSED=0

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[fail] neither python3 nor python was found"
    exit 1
  fi
fi

mark_failed() {
  FAILED=1
}

mark_basic_failed() {
  BASIC_CHECKS_PASSED=0
  mark_failed
}

mark_core_failed() {
  CORE_API_PASSED=0
  mark_basic_failed
}

print_section() {
  printf '\n== %s ==\n' "$1"
}

assert_body_contains() {
  local file="$1"
  local expected="$2"
  local label="$3"

  if grep -Fq "$expected" "$file"; then
    echo "[ok] ${label}"
  else
    echo "[fail] ${label}: expected response to contain ${expected}"
    mark_basic_failed
    return 1
  fi
  return 0
}

check_json_endpoint() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local body_file
  local status_file
  local status
  local core_endpoint=0

  if [[ "$path" == "/health" || "$path" == "/ready" ]]; then
    core_endpoint=1
  fi

  body_file="$(mktemp)"
  status_file="$(mktemp)"

  print_section "$label"
  status="$(
    curl "${CURL_TLS_ARGS[@]}" \
      --silent --show-error \
      --max-time "$TIMEOUT_SEC" \
      --output "$body_file" \
      --write-out "%{http_code}" \
      "${BASE_URL}${path}" 2>"$status_file"
  )"
  local curl_rc=$?

  if [[ "$curl_rc" -ne 0 ]]; then
    echo "[fail] ${label}: curl failed"
    sed -n '1,120p' "$status_file"
    if [[ "$core_endpoint" -eq 1 ]]; then
      mark_core_failed
    else
      mark_basic_failed
    fi
  else
    echo "HTTP ${status}"
    sed -n '1,120p' "$body_file"
    echo
    if [[ "$status" != "200" ]]; then
      echo "[fail] ${label}: expected HTTP 200"
      if [[ "$core_endpoint" -eq 1 ]]; then
        mark_core_failed
      else
        mark_basic_failed
      fi
    fi
    if ! assert_body_contains "$body_file" "$expected" "$label body"; then
      if [[ "$core_endpoint" -eq 1 ]]; then
        CORE_API_PASSED=0
      fi
    fi
  fi

  rm -f "$body_file" "$status_file"
}

check_tls_certificate() {
  if [[ "$BASE_URL" != https://* ]]; then
    return 0
  fi

  print_section "TLS certificate"
  if curl --silent --show-error --max-time "$TIMEOUT_SEC" --head "${BASE_URL}/health" >/dev/null; then
    echo "[ok] TLS certificate is trusted by curl"
  else
    echo "[fail] TLS certificate is not trusted by curl"
    echo "       Re-run with INSECURE=1 only for proxy diagnostics."
    mark_failed
  fi
}

check_docs() {
  local status

  print_section "GET /docs"
  status="$(
    curl "${CURL_TLS_ARGS[@]}" \
      --silent --show-error \
      --max-time "$TIMEOUT_SEC" \
      --output /dev/null \
      --write-out "%{http_code}" \
      --head \
      "${BASE_URL}/docs"
  )"
  local curl_rc=$?

  if [[ "$curl_rc" -ne 0 ]]; then
    echo "[fail] /docs request failed"
    mark_basic_failed
    return
  fi

  echo "HTTP ${status}"
  if [[ "$status" == "200" ]]; then
    echo "[ok] /docs loads"
  else
    echo "[fail] /docs expected HTTP 200"
    mark_basic_failed
  fi
}

check_legacy_infer() {
  local body_file
  local status

  body_file="$(mktemp)"
  print_section "POST /api/infer legacy endpoint"
  status="$(
    curl "${CURL_TLS_ARGS[@]}" \
      --silent --show-error \
      --max-time "$TIMEOUT_SEC" \
      --output "$body_file" \
      --write-out "%{http_code}" \
      -X POST "${BASE_URL}/api/infer" \
      -H "Content-Type: application/json" \
      -d '{"encrypted_embeddings":[],"metadata":{}}'
  )"
  local curl_rc=$?

  if [[ "$curl_rc" -ne 0 ]]; then
    echo "[fail] /api/infer request failed"
    mark_basic_failed
  else
    echo "HTTP ${status}"
    sed -n '1,120p' "$body_file"
    echo
    if [[ "$status" != "200" ]]; then
      echo "[fail] /api/infer expected HTTP 200"
      mark_basic_failed
    fi
    assert_body_contains "$body_file" "Use /api/session + /api/layer for real inference" "/api/infer body" || true
  fi

  rm -f "$body_file"
}

check_real_session() {
  print_section "POST /api/session with real CKKS public context"
  BASE_URL="$BASE_URL" INSECURE="$INSECURE" TIMEOUT_SEC="$TIMEOUT_SEC" PYTHONWARNINGS="${PYTHONWARNINGS:-}" "$PYTHON_BIN" - <<'PY'
import base64
import os
import sys

import requests

from client.encryption.ckks_context import create_ckks_context, serialize_public_context

base_url = os.environ["BASE_URL"].rstrip("/")
timeout = int(os.environ.get("TIMEOUT_SEC", "60"))
verify = os.environ.get("INSECURE", "0") != "1"

ctx = create_ckks_context()
public_b64 = base64.b64encode(serialize_public_context(ctx)).decode("utf-8")

try:
    response = requests.post(
        f"{base_url}/api/session",
        json={"public_context_b64": public_b64},
        timeout=timeout,
        verify=verify,
    )
except requests.RequestException as exc:
    print(f"[fail] session request failed: {exc}")
    sys.exit(1)

print(f"HTTP {response.status_code}")
print(response.text[:500])

if response.status_code != 200:
    sys.exit(1)

session_id = response.json().get("session_id")
if not session_id:
    print("[fail] response did not include session_id")
    sys.exit(1)

print(f"[ok] session_id: {session_id}")
PY
  local py_rc=$?

  if [[ "$py_rc" -ne 0 ]]; then
    mark_failed
  else
    SESSION_CHECK_PASSED=1
  fi
}

run_full_client() {
  print_section "Full client generation"
  ZKLLM_SERVER_BASE_URL="$BASE_URL" \
    "$PYTHON_BIN" -m client.client \
      --prompt "$PROMPT" \
      --num-tokens "$NUM_TOKENS" \
      --num-encrypted-layers "$NUM_ENCRYPTED_LAYERS" \
      --stats
  local client_rc=$?

  if [[ "$client_rc" -ne 0 ]]; then
    mark_failed
  fi
}

echo "Testing deployed API at: ${BASE_URL}"
if [[ "$INSECURE" == "1" ]]; then
  echo "TLS verification: disabled for diagnostics"
else
  echo "TLS verification: enabled"
fi
echo "Python executable: ${PYTHON_BIN}"

check_tls_certificate
check_json_endpoint "/health" '"status":"ok"' "GET /health"
check_json_endpoint "/ready" '"status":"ready"' "GET /ready"
check_docs
check_legacy_infer

if [[ "$CHECK_SESSION" == "1" && "$CORE_API_PASSED" -eq 1 ]]; then
  check_real_session
elif [[ "$CHECK_SESSION" == "1" ]]; then
  echo
  echo "Skipping /api/session check because /health or /ready failed."
else
  echo
  echo "Skipping /api/session check because CHECK_SESSION=0"
fi

if [[ "$RUN_CLIENT" == "1" && "$BASIC_CHECKS_PASSED" -eq 1 && "$SESSION_CHECK_PASSED" -eq 1 ]]; then
  run_full_client
elif [[ "$RUN_CLIENT" == "1" ]]; then
  echo
  echo "Skipping full client generation because prerequisite checks failed."
else
  echo
  echo "Skipping full client generation. Set RUN_CLIENT=1 to run it."
fi

print_section "Summary"
if [[ "$FAILED" -eq 0 ]]; then
  echo "[ok] Deployment smoke test passed."
  exit 0
fi

echo "[fail] Deployment smoke test failed."
echo "If every HTTPS request says 'no available server', fix the deployment route/container before rerunning."
exit 1
