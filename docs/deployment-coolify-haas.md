# ZL-LLM Coolify Runtime + HAAS Daemon Deployment

This note captures the target production deployment where Coolify owns the
ZL-LLM FastAPI runtime and the publisher platform HAAS owns only the
marketplace-facing daemon.

## Endpoint Map

| Component | Value |
| --- | --- |
| Coolify service URL | `https://zllm-service.example.com` |
| Coolify app port | `8000` |
| Health endpoint | `GET /health` |
| Readiness endpoint | `GET /ready` |
| Session endpoint | `POST /api/session` |
| Layer endpoint | `POST /api/layer` |
| Legacy infer endpoint | `POST /api/infer` |
| HAAS daemon ID | `2b4d4f3b-c043-4671-ad6a-697012038574` |
| Hosted service ID to replace | `cb4c9cfb-bbf9-49b2-8357-d18926a7a5df` |
| Organization ID | `TODO` |
| Service ID/name | `TODO` |
| Group ID/name | `TODO` |
| Network | `TODO` |
| Metadata URI/IPFS hash | `TODO` |

## Coolify Application Settings

- Build pack: Dockerfile
- Runtime image: `linux/amd64` `python:3.9-slim`
- Repository/branch: `TODO`
- Port Exposes: `8000`
- Domain: `https://zllm-service.example.com`
- HTTPS: enabled
- Health check path: `/ready`
- Expected health status: `200`
- Persistent volume: mount Hugging Face cache at `/root/.cache/huggingface`

Required environment variables:

```bash
HOST=0.0.0.0
PORT=8000
ZKLLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
HF_HOME=/root/.cache/huggingface
PYTHONUNBUFFERED=1
ZKLLM_LOG_LEVEL=INFO
```

Optional environment variables:

```bash
TRANSFORMERS_CACHE=/root/.cache/huggingface
```

The Dockerfile uses Python 3.9 and forces `linux/amd64` because the current
`tenseal==0.3.16` package resolved in the project virtualenv on Python 3.9,
but did not resolve during Docker builds for Python 3.11 or Linux ARM64.
Run this service on an AMD64 Coolify host unless TenSEAL is replaced or built
from source for another platform.

## Coolify Verification

After deployment, verify the service directly:

```bash
curl -i https://zllm-service.example.com/health
curl -i https://zllm-service.example.com/ready
```

Expected behavior:

- `/health` returns `200` with `{"status":"ok"}`.
- `/ready` returns `200` only after TinyLlama is loaded.
- Coolify logs show Uvicorn listening on `0.0.0.0:8000`.
- Coolify logs show `Model loaded, server ready.`

Run a direct client smoke test:

```bash
ZKLLM_SERVER_BASE_URL=https://zllm-service.example.com \
python -m client.client \
  --prompt "The capital of France is" \
  --num-tokens 2 \
  --num-encrypted-layers 1 \
  --stats
```

## HAAS Daemon Redeploy

Before changing the publisher platform, capture:

- Organization ID
- Service ID/name
- Group ID/name
- Network
- Current daemon endpoint
- Current hosted service endpoint
- Pricing/payment group details
- Metadata URI/IPFS hash
- Marketplace offline screenshot with timestamp and URL

Redeploy only the daemon:

1. Do not create a new HAAS hosted service runtime.
2. Configure the daemon backend/service endpoint as:
   ```text
   https://zllm-service.example.com
   ```
3. Confirm the daemon config still uses the existing organization, service,
   group, network, and payment settings.
4. Deploy the daemon and wait for HAAS status `UP`.
5. Test daemon-to-Coolify forwarding with the smallest supported request.

## Metadata Validation

After daemon redeploy:

1. Inspect published service metadata.
2. Confirm the group daemon endpoint maps to the new HAAS daemon endpoint.
3. Confirm no metadata still points to the old HAAS hosted service runtime.
4. Republish metadata only if endpoint or group data changed.
5. Record the new metadata URI/IPFS hash in this file.
6. Wait for marketplace cache/index refresh before judging online status.

## Local Verification

Quick tests:

```bash
pytest -q
```

Server endpoint tests:

```bash
pytest server/tests/test_api_endpoint.py -q
```

Slow end-to-end test where model download is available:

```bash
pytest -m slow server/tests/test_integration_e2e.py -v
```

Container smoke test:

```bash
docker build -t zllm-service .
docker run --rm -p 8000:8000 zllm-service
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

## Failure Handling

If Coolify returns `404` or `No available server`:

- Check Coolify health status.
- Confirm `Port Exposes` is `8000`.
- Confirm `/ready` returns `200` inside the container.
- Confirm TinyLlama finished loading.

If HAAS daemon is `UP` but marketplace is offline:

1. Check marketplace metadata daemon endpoint.
2. Check daemon public reachability.
3. Check daemon backend URL is the Coolify URL.
4. Check Coolify logs during a marketplace request.
5. Check payment group and group ID alignment.
6. Allow for marketplace cache/index delay.

If HAAS cannot forward HTTP:

- Stop the cutover.
- Keep Coolify deployed.
- Add a gRPC/SingularityNET adapter plan around the current FastAPI runtime.
- Do not delete the old HAAS hosted service.

## Rollback

1. Preserve Coolify logs and HAAS daemon logs.
2. Repoint daemon backend to the previous HAAS hosted service if supported.
3. Otherwise redeploy the previous full-HAAS setup from the captured config.
4. Keep current metadata unchanged unless a known-good replacement is ready.
5. Disable, but do not delete, the Coolify app until rollback is verified.
