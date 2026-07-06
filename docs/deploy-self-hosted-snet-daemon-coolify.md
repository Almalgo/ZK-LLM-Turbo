# Self-Hosted SNET Daemon On Coolify

This deployment replaces HaaS Daemon Only with a self-hosted `snetd` container.
Coolify hosts the daemon and the existing Coolify FastAPI app remains the
service backend.

## Target

```text
Marketplace / client
        |
        v
https://daemon-zkllm.almalgo.com
        |
        v
snetd container on Coolify
        |
        v
https://zkllm.almalgo.com
```

## Coolify application

Use this repo as the source, not the upstream `singnet/snet-daemon` repo. The
wrapper Dockerfile downloads the official `snetd` release binary and renders the
daemon config from environment variables.

```text
Repository URL: https://github.com/Almalgo/ZK-LLM-Turbo
Branch: main
Build Pack: Dockerfile
Base Directory: /
Dockerfile Location: /Dockerfile.snetd
Port: 7000
Is it a static site?: No
Domain: https://daemon-zkllm.almalgo.com
```

Set Coolify persistent storage:

```text
/data/snetd
```

## Required environment variables

```env
SNET_ORG_ID=almalgo_labs
SNET_SERVICE_ID=zk_llm3
SNET_DAEMON_GROUP_NAME=default_group
SNET_DAEMON_PORT=7000
SNET_BACKEND_URL=https://zkllm.almalgo.com
SNET_HEARTBEAT_URL=https://zkllm.almalgo.com/heartbeat
SNET_ETHEREUM_JSON_RPC_HTTP_ENDPOINT=<mainnet_rpc_http_url>
SNET_TOKEN_SECRET_KEY=<random_32_plus_character_secret>
SNET_LOG_LEVEL=debug
```

Optional:

```env
SNET_PRIVATE_KEY_FOR_FREE_CALLS=
SNET_SERVICE_TIMEOUT=15m
SNET_MAX_MESSAGE_SIZE_MB=32
SNET_ETCD_DATA_DIR=/data/snetd/etcd
```

Do not put wallet owner private keys in this daemon unless a specific SNET free
call signing flow requires it. Service metadata publishing still happens through
the local `snet` CLI and owner wallet.

## Service metadata endpoint

When publishing the new service, use:

```text
https://daemon-zkllm.almalgo.com
```

If Coolify requires the port in the public URL, use:

```text
https://daemon-zkllm.almalgo.com:7000
```

The metadata endpoint must point to the daemon, not directly to the FastAPI
backend.

## Verification

Backend:

```bash
curl -i https://zkllm.almalgo.com/heartbeat
curl -i -X POST https://zkllm.almalgo.com/health \
  -H 'content-type: application/json' \
  -d '{"op":"health"}'
```

Daemon:

```bash
curl -i https://daemon-zkllm.almalgo.com/heartbeat
```

Expected daemon logs:

```text
service_endpoint = https://zkllm.almalgo.com
service_heartbeat_type = http
heartbeat_endpoint = https://zkllm.almalgo.com/heartbeat
Daemon successfully started
```

Bad logs:

```text
serviceHeartbeatURL: ""
Get "": unsupported protocol scheme ""
```
