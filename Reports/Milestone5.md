# Milestone 5 Resubmission: Public Service Availability and Reliability Evidence

Date: 2026-05-17

Project: Zero-Knowledge Secure LLM

Milestone: 5

## 1. Resubmission Summary

The previous Milestone 5 submission was rejected because the public SingularityNET service was offline at review time and the reliability evidence did not sufficiently demonstrate stable successful operation.

The deployment has now been corrected for SingularityNET Publisher Hosting-as-a-Service (HaaS). The repository was reformatted to match the Full-Stack HaaS service contract, and the service health/profile path now returns the expected serving status from the HaaS handler.

Current corrected evidence is presented below.

## 2. Public Service Identifiers

Organization ID:

```text
almalgo_labs
```

Service ID:

```text
zk_llm1
```

Marketplace service page:

```text
https://marketplace.singularitynet.io/servicedetails/org/almalgo_labs/service/zk_llm1/tab/0
```

Daemon endpoint:

```text
https://1cf76242e956da6397b473a57f2ca4a9245568fe90b76910c1b18b4f.mainnet.haas.singularitynet.dev
```

Hosted service heartbeat endpoint:

```text
https://api.haas.singularitynet.dev/orchestrator/v1/services/almalgo_labs/zk_llm1/heartbeat
```

Daemon ID:

```text
2b4d4f3b-c043-4671-ad6a-697012038574
```

Hosted Service ID:

```text
cb4c9cfb-bbf9-49b2-8357-d18926a7a5df
```

Latest corrected commit:

```text
d3ed214e29de9e0f4606913235fe8300efe54929
Simplify health check response
```

## 3. Publisher Status Evidence

The Publisher HaaS dashboard screenshot shows both managed components running:

```text
Daemon Status: UP
Hosted Service Status: UP
Daemon Last Modified Date: 19:51:44, May 17, 2026
Hosted Service Last Modified Date: 19:46:31, May 17, 2026
```

Screenshot artifact to include with this report:

```text
evidence/m5_publisher_haas_components_up_2026-05-17.png
```

![Publisher HaaS components UP](../evidence/m5_publisher_haas_components_up_2026-05-17.png)

## 4. Hosted Service Health Evidence

The HaaS Hosted Service logs show that the managed platform invoked the service health operation and the service returned the expected serving response.

Log excerpt:

```text
2026-05-17 18:40:02.108 [handler] [INFO] Handler started | {"input_data":{"op":"health"}}
2026-05-17 18:40:02.194 [handler] [INFO] Run completed successfully | {"result_type":"dict","result_preview":"{'serviceID': 'zk_llm1', 'status': 'SERVING'}"}
```

Expected HaaS health response:

```json
{
  "serviceID": "zk_llm1",
  "status": "SERVING"
}
```

This health operation is intentionally lightweight and does not load the TinyLlama model. It validates service availability without triggering model download or encrypted inference.

## 5. Repository Corrections Since Rejection

The repository now follows the SingularityNET Full-Stack HaaS layout:

```text
customer_main.py
requirements.txt
profile.json
runpod_handler.py
Dockerfile
```

The HaaS handler entrypoint is:

```python
def run(input_data: dict):
    ...
```

Supported HaaS operations:

```text
health
heartbeat
session
layer
```

The health and heartbeat operations now return:

```json
{"serviceID":"zk_llm1","status":"SERVING"}
```

The FastAPI service is still available for local development and external HTTP testing through:

```text
GET  /
GET  /heartbeat
GET  /health
POST /api/session
POST /api/layer
```

## 6. Local Reproduction: HaaS Handler

Clone the repository and check out the corrected commit:

```bash
git clone https://github.com/Almalgo/ZK-LLM-Turbo.git
cd ZK-LLM-Turbo
git checkout d3ed214e29de9e0f4606913235fe8300efe54929
```

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the lightweight HaaS health check locally:

```bash
python3 - <<'PY'
from customer_main import run
print(run({"op": "health"}))
PY
```

Expected output:

```text
{'serviceID': 'zk_llm1', 'status': 'SERVING'}
```

Run the HaaS handler tests:

```bash
pytest tests/test_customer_main.py
```

Observed result on 2026-05-17:

```text
5 passed
```

## 7. Local Reproduction: FastAPI Service

Build the FastAPI local-development container:

```bash
docker buildx build --load -f Dockerfile.fastapi -t zk-llm-turbo-fastapi .
```

Run the container:

```bash
docker run --rm -p 8000:8000 zk-llm-turbo-fastapi
```

In another terminal, verify liveness endpoints:

```bash
curl -i http://127.0.0.1:8000/
curl -i http://127.0.0.1:8000/heartbeat
curl -i http://127.0.0.1:8000/health
```

Expected result for each:

```text
HTTP/1.1 200 OK
```

Run the endpoint preflight check:

```bash
python3 scripts/snet_endpoint_preflight.py \
  --public-base-url http://127.0.0.1:8000 \
  --allow-http
```

Expected result:

```text
PASS ... /
PASS ... /heartbeat
PASS ... /health
```

Run the API tests:

```bash
pytest server/tests/test_api_endpoint.py
```

Observed result on 2026-05-17:

```text
6 passed
```

## 8. Local Test Evidence

Combined local test command:

```bash
pytest tests/test_customer_main.py server/tests/test_api_endpoint.py
```

Observed result on 2026-05-17:

```text
11 passed
```

This confirms:

- The HaaS handler health path returns `SERVING`.
- Missing/invalid HaaS operations return structured errors.
- Existing layer operation dispatch remains intact.
- FastAPI `/`, `/heartbeat`, and `/health` routes are available.
- FastAPI model-load failures are handled as HTTP 503 rather than startup crashes.

## 9. Functional Verification Path

For independent validation through SingularityNET, reviewers should use the Marketplace service page:

```text
https://marketplace.singularitynet.io/servicedetails/org/almalgo_labs/service/zk_llm1/tab/0
```

The deployed service is backed by HaaS:

```text
Daemon: UP
Hosted Service: UP
```

The hosted handler health operation has been verified in service logs:

```text
op=health -> {'serviceID': 'zk_llm1', 'status': 'SERVING'}
```

## 10. Reviewer Instructions

1. Open the Marketplace service page:

```text
https://marketplace.singularitynet.io/servicedetails/org/almalgo_labs/service/zk_llm1/tab/0
```

2. Confirm the organization and service:

```text
Organization ID: almalgo_labs
Service ID: zk_llm1
```

3. Confirm the HaaS deployment status using the attached Publisher screenshot:

```text
Daemon: UP
Hosted Service: UP
```

4. Reproduce the local HaaS health contract:

```bash
python3 - <<'PY'
from customer_main import run
print(run({"op": "health"}))
PY
```

Expected:

```text
{'serviceID': 'zk_llm1', 'status': 'SERVING'}
```

5. Reproduce local FastAPI liveness:

```bash
docker buildx build --load -f Dockerfile.fastapi -t zk-llm-turbo-fastapi .
docker run --rm -p 8000:8000 zk-llm-turbo-fastapi
curl -i http://127.0.0.1:8000/
curl -i http://127.0.0.1:8000/heartbeat
curl -i http://127.0.0.1:8000/health
```

Expected:

```text
HTTP 200 for all three liveness endpoints
```

## 11. Evidence Artifacts

Current corrected evidence:

```text
Reports/Milestone5_Resubmission.md
evidence/m5_publisher_haas_components_up_2026-05-17.png
Hosted Service log excerpt showing op=health -> SERVING
Commit d3ed214e29de9e0f4606913235fe8300efe54929
pytest result: tests/test_customer_main.py -> 5 passed
pytest result: server/tests/test_api_endpoint.py -> 6 passed
```

Superseded evidence from the original rejected submission should not be used as current proof.

## 12. Notes On Managed HaaS Health Propagation

The Publisher screenshot and Hosted Service logs show that the HaaS-managed components and service handler are running. If the Marketplace status display lags behind the Publisher status, the next operational step is to request SNET support to refresh or repair the managed HaaS orchestrator health state for:

```text
Organization ID: almalgo_labs
Service ID: zk_llm1
Hosted Service ID: cb4c9cfb-bbf9-49b2-8357-d18926a7a5df
Daemon ID: 2b4d4f3b-c043-4671-ad6a-697012038574
```

