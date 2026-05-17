# Milestone 6 Resubmission: MVP And Public Validation Evidence

Date: 2026-05-17

Deep Funding Round: 4

Project: Zero-Knowledge Secure LLM

Milestone: 6

Deliverable: Milestone 6 - MVP & Tests (v0.2)

## 1. Resubmission Summary

The previous Milestone 6 submission was rejected because the milestone outcome could not be sufficiently verified. The rejection identified three verification gaps:

```text
1. Functionally verifiable MVP not sufficiently demonstrated.
2. Public testing activities were not evidenced.
3. Operational product validation remained unclear.
```

This updated report addresses those issues by providing:

```text
1. Publicly accessible codebase and notebook.
2. Step-by-step local MVP reproduction.
3. SNET Publisher/HaaS service status evidence.
4. Hosted service health logs.
5. Clarification on public testing expectations and DF meeting availability.
```

This report does not claim that a live Deep Funding town hall test session occurred. Instead, it documents the current public validation path that reviewers can reproduce independently from the published repository, notebook, local service, and HaaS deployment evidence.

## 2. MVP Definition

The Milestone 6 MVP is:

```text
A privacy-preserving split-inference prototype where the client creates CKKS-encrypted vectors, the server performs supported homomorphic linear-layer operations, and the client retains secret-key control for decryption and non-linear operations.
```

The MVP product surfaces are:

```text
1. Public GitHub codebase.
2. Public Jupyter notebook.
3. Local FastAPI service.
4. SNET Publisher/HaaS hosted service.
5. Marketplace demo package.
```

The notebook demonstrates the privacy and computation model. The local service and HaaS handler demonstrate the operational service shape expected by reviewers and the SNET platform.

## 3. Public Resources

Code base:

```text
https://github.com/Almalgo/ZK-LLM-Turbo
```

Public notebook:

```text
https://github.com/Almalgo/ZK-LLM-Turbo/blob/main/notebooks/milestone6-demo.ipynb
```

Final report:

```text
https://github.com/Almalgo/ZK-LLM-Turbo/blob/main/Reports/Milestone6.md
```

Marketplace service:

```text
https://marketplace.singularitynet.io/servicedetails/org/almalgo_labs/service/zk_llm1/tab/0
```

Marketplace demo package in repository:

```text
demo/zk-llm-turbo-dapp-demo.zip
```

## 4. Independent Local MVP Validation

Reviewers can reproduce the HaaS MVP handler locally from the public repository.

Clone the repository:

```bash
git clone https://github.com/Almalgo/ZK-LLM-Turbo.git
cd ZK-LLM-Turbo
```

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the HaaS handler health path:

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

Run the focused MVP/HaaS and API tests:

```bash
pytest tests/test_customer_main.py server/tests/test_api_endpoint.py
```

Expected current result:

```text
11 passed
```

Run the public notebook:

```bash
jupyter notebook notebooks/milestone6-demo.ipynb
```

Notebook validation criteria:

```text
1. CKKS context is created client-side.
2. Public context can be serialized for server use.
3. Server-side homomorphic matrix multiplication is demonstrated.
4. Client-side non-linear operations are demonstrated.
5. Timing breakdown is produced.
```

## 5. Local FastAPI MVP Validation

The local HTTP product path can be reproduced with the FastAPI container.

Build the FastAPI local-development image:

```bash
docker buildx build --load -f Dockerfile.fastapi -t zk-llm-turbo-fastapi .
```

Run the container:

```bash
docker run --rm -p 8000:8000 zk-llm-turbo-fastapi
```

Check liveness endpoints from another terminal:

```bash
curl -i http://127.0.0.1:8000/
curl -i http://127.0.0.1:8000/heartbeat
curl -i http://127.0.0.1:8000/health
```

Expected result:

```text
HTTP 200 for all three endpoints
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

## 6. Public Hosted Service Evidence

SNET/HaaS identifiers:

```text
Organization ID: almalgo_labs
Service ID: zk_llm1
Daemon ID: 2b4d4f3b-c043-4671-ad6a-697012038574
Hosted Service ID: cb4c9cfb-bbf9-49b2-8357-d18926a7a5df
```

The Publisher HaaS dashboard screenshot shows both managed components as UP:

```text
Daemon: UP
Hosted Service: UP
```

Screenshot artifact:

```text
evidence/m6_publisher_service_up_2026-05-17.png
```

Hosted Service health log excerpt:

```text
2026-05-17 18:40:02.108 [handler] [INFO] Handler started | {"input_data":{"op":"health"}}
2026-05-17 18:40:02.194 [handler] [INFO] Run completed successfully | {"result_type":"dict","result_preview":"{'serviceID': 'zk_llm1', 'status': 'SERVING'}"}
```

This proves that the HaaS-managed service handler receives health calls and returns the expected serving response:

```json
{"serviceID":"zk_llm1","status":"SERVING"}
```

If a Marketplace status display lags behind Publisher/HaaS status, the report should be evaluated together with the Publisher screenshot, Hosted Service logs, and local reproduction evidence.

## 7. Public Testing Activities And Deep Funding Meeting Availability

The project publicly shared the codebase, notebook, report, and service resources for public use and independent validation. The project did not commit to recruiting a fixed tester cohort or running a formal usability study.

The originally referenced Deep Funding community meeting / town hall channel is no longer available in the same form, so a live DF public testing session could not be scheduled for this resubmission.

Because that venue is unavailable, the resubmission provides objective public validation materials instead:

```text
- public repository,
- public notebook,
- SNET Marketplace service page,
- local reproduction commands,
- Publisher/HaaS status screenshot,
- Hosted Service health logs,
- test commands and expected outputs.
```

No claim is made that a Deep Funding town hall or public community feedback meeting occurred.

## 8. Feedback Collection Status

No formal public tester feedback dataset is submitted because the milestone deliverable was interpreted as publishing public resources for use and validation, not as conducting a structured user research study.

The public validation path is therefore based on objective reproducibility:

```text
- reviewers can clone the code,
- run the notebook,
- run the HaaS handler health check,
- run the FastAPI service locally,
- execute the test suite,
- inspect the SNET Publisher/HaaS evidence.
```

If Deep Funding requires a formal feedback dataset for reassessment, the team can run a follow-up public feedback round once an available community channel is confirmed.

## 9. Evidence Matrix

| Rejection Item | Corrected Evidence | Status |
|---|---|---|
| MVP not independently verifiable | Local setup, notebook, Docker FastAPI, HaaS handler commands | Addressed |
| Public testing activities not evidenced | Public resource sharing documented; DF public meeting unavailability clarified | Clarified |
| Feedback not documented | No formal tester dataset claimed; reproducibility evidence provided instead | Clarified |
| Operational validation unclear | Publisher UP screenshot and HaaS health logs | Addressed |
| Notebook too internal | Notebook is now paired with explicit MVP/product reproduction instructions | Addressed |

## 10. Public APIs And Interfaces

No code API changes are required for this report update.

Existing public surfaces remain:

```text
customer_main.run({"op": "health"})
customer_main.run({"op": "heartbeat"})
GET  /
GET  /heartbeat
GET  /health
POST /api/session
POST /api/layer
```

Expected HaaS health response:

```json
{"serviceID":"zk_llm1","status":"SERVING"}
```

FastAPI liveness response examples:

```text
GET /          -> {"status":"ok","service":"zk-llm-turbo"}
GET /heartbeat -> {"status":"ok"}
GET /health    -> {"status":"ok", ...}
```

## 11. Test Cases And Scenarios

Focused test gate:

```bash
pytest tests/test_customer_main.py server/tests/test_api_endpoint.py
```

Expected:

```text
11 passed
```

Optional broader test gate:

```bash
pytest -m "not slow"
```

Expected based on prior report:

```text
152 passed
```

Notebook validation:

```bash
jupyter notebook notebooks/milestone6-demo.ipynb
```

Expected:

```text
All cells run successfully and demonstrate CKKS privacy, HE matmul, client-side non-linear ops, and timing breakdown.
```

FastAPI local validation:

```bash
docker buildx build --load -f Dockerfile.fastapi -t zk-llm-turbo-fastapi .
docker run --rm -p 8000:8000 zk-llm-turbo-fastapi
curl -i http://127.0.0.1:8000/
curl -i http://127.0.0.1:8000/heartbeat
curl -i http://127.0.0.1:8000/health
```

Expected:

```text
HTTP 200 for all liveness routes
```

## 12. Acceptance Criteria For Resubmission

The revised report is complete when:

```text
1. Reports/Milestone6.md is updated in place.
2. It includes public links to code, notebook, report, and Marketplace service.
3. It includes step-by-step local reproduction commands.
4. It includes expected outputs.
5. It includes Publisher/HaaS UP screenshot reference.
6. It includes Hosted Service health log excerpt.
7. It directly explains DF meeting/town hall unavailability.
8. It does not falsely claim public testing sessions or feedback collection.
9. It clearly states what was and was not completed.
```

## 13. Evidence Artifacts

Evidence artifacts referenced by this report:

```text
evidence/m6_publisher_service_up_2026-05-17.png
evidence/m6_hosted_service_health_log_2026-05-17.txt
evidence/m6_local_test_output_2026-05-17.txt
evidence/m6_notebook_run_screenshot_2026-05-17.png
```

The Hosted Service health log is quoted inline in this report so reviewers can evaluate the service status even if the raw transcript is submitted separately.

## 14. Explicit Assumptions And Defaults

```text
- The report update uses the existing filename: Reports/Milestone6.md.
- No new Reports/Milestone6_Resubmission.md file is used.
- The primary MVP proof is notebook + local reproduction.
- Public DF community meeting/town hall evidence is not fabricated.
- The report explicitly states that those meetings are no longer available.
- The report clarifies that public resource sharing was the committed public availability activity.
- The Publisher/HaaS screenshot and service health logs are used as operational product evidence.
- Any Marketplace/offline propagation issue should not be hidden, but the report focuses on objective MVP reproducibility and hosted-handler health evidence.
```

## 15. Version

v0.2 - MVP Release Resubmission

