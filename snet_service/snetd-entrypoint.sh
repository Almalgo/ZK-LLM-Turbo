#!/bin/sh
set -eu

require_env() {
  name="$1"
  eval "value=\${$name:-}"
  if [ -z "$value" ]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_env SNET_ORG_ID
require_env SNET_SERVICE_ID
require_env SNET_ETHEREUM_JSON_RPC_HTTP_ENDPOINT

SNET_DAEMON_PORT="${SNET_DAEMON_PORT:-7000}"
SNET_DAEMON_GROUP_NAME="${SNET_DAEMON_GROUP_NAME:-default_group}"
SNET_BACKEND_URL="${SNET_BACKEND_URL:-https://zkllm.almalgo.com}"
SNET_HEARTBEAT_URL="${SNET_HEARTBEAT_URL:-https://zkllm.almalgo.com/heartbeat}"
SNET_IPFS_ENDPOINT="${SNET_IPFS_ENDPOINT:-https://ipfs.singularitynet.io:443}"
SNET_SERVICE_TIMEOUT="${SNET_SERVICE_TIMEOUT:-15m}"
SNET_MAX_MESSAGE_SIZE_MB="${SNET_MAX_MESSAGE_SIZE_MB:-32}"
SNET_TRUSTED_FREE_CALL_SIGNERS="${SNET_TRUSTED_FREE_CALL_SIGNERS:-\"0x3Bb9b2499c283cec176e7C707Ecb495B7a961ebf\", \"0x7DF35C98f41F3Af0df1dc4c7F7D4C19a71Dd059F\"}"
SNET_TOKEN_SECRET_KEY="${SNET_TOKEN_SECRET_KEY:-change-this-token-secret-key-before-production}"
SNET_ETCD_DATA_DIR="${SNET_ETCD_DATA_DIR:-/data/snetd/etcd}"

mkdir -p "$SNET_ETCD_DATA_DIR"
chmod 700 "$SNET_ETCD_DATA_DIR" 2>/dev/null || true

cat > /app/snetd.config.json <<EOF
{
  "blockchain_enabled": true,
  "blockchain_network_selected": "main",
  "organization_id": "${SNET_ORG_ID}",
  "service_id": "${SNET_SERVICE_ID}",
  "daemon_group_name": "${SNET_DAEMON_GROUP_NAME}",
  "daemon_endpoint": "0.0.0.0:${SNET_DAEMON_PORT}",
  "daemon_type": "grpc",
  "passthrough_enabled": true,
  "service_type": "http",
  "service_endpoint": "${SNET_BACKEND_URL}",
  "service_heartbeat_type": "http",
  "heartbeat_endpoint": "${SNET_HEARTBEAT_URL}",
  "service_timeout": "${SNET_SERVICE_TIMEOUT}",
  "max_message_size_in_mb": ${SNET_MAX_MESSAGE_SIZE_MB},
  "ipfs_endpoint": "${SNET_IPFS_ENDPOINT}",
  "ethereum_json_rpc_http_endpoint": "${SNET_ETHEREUM_JSON_RPC_HTTP_ENDPOINT}",
  "payment_channel_storage_type": "etcd",
  "payment_channel_storage_client": {
    "connection_timeout": "5s",
    "request_timeout": "3s",
    "endpoints": ["http://127.0.0.1:2379"]
  },
  "payment_channel_storage_server": {
    "enabled": true,
    "id": "storage-1",
    "scheme": "http",
    "host": "127.0.0.1",
    "client_port": 2379,
    "peer_port": 2380,
    "token": "zkllm3-snetd-storage",
    "cluster": "storage-1=http://127.0.0.1:2380",
    "data_dir": "${SNET_ETCD_DATA_DIR}",
    "startup_timeout": "1m",
    "log_level": "warn",
    "log_outputs": ["stdout"]
  },
  "private_key_for_free_calls": "${SNET_PRIVATE_KEY_FOR_FREE_CALLS:-}",
  "trusted_free_call_signers": [${SNET_TRUSTED_FREE_CALL_SIGNERS}],
  "token_secret_key": "${SNET_TOKEN_SECRET_KEY}",
  "log": {
    "level": "${SNET_LOG_LEVEL:-debug}",
    "output": {
      "type": ["stdout"]
    }
  }
}
EOF

echo "Starting snetd for ${SNET_ORG_ID}/${SNET_SERVICE_ID} on 0.0.0.0:${SNET_DAEMON_PORT}"
exec snetd --config /app/snetd.config.json
