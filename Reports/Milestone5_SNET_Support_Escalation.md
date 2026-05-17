# Milestone 5 SNET Support Escalation

Date: 2026-05-17

## Summary

Publisher UI shows both HaaS components as `UP`, but public endpoint checks still
show the service as unavailable to external reviewers.

This blocks Milestone 5 resubmission because the rejection specifically cited
that the public service was not functionally verifiable.

## Service Identifiers

```text
Organization ID: almalgo_labs
Service ID: zk_llm1
Daemon ID: 2b4d4f3b-c043-4671-ad6a-697012038574
Hosted Service ID: cb4c9cfb-bbf9-49b2-8357-d18926a7a5df
Latest deployed repo commit expected: ea240b7cff4e1a4f5ce8549e0ea3ba95a7ed468c
```

## Public URLs

```text
Marketplace:
https://marketplace.singularitynet.io/servicedetails/org/almalgo_labs/service/zk_llm1/tab/0

Daemon:
https://1cf76242e956da6397b473a57f2ca4a9245568fe90b76910c1b18b4f.mainnet.haas.singularitynet.dev

Hosted Service heartbeat:
https://api.haas.singularitynet.dev/orchestrator/v1/services/almalgo_labs/zk_llm1/heartbeat
```

## Observed Public Check Failure

Captured artifact:

```text
benchmarks/results/m5_public_service_evidence_2026-05-17.json
```

Command:

```bash
python3 scripts/m5_public_service_evidence.py \
  --output benchmarks/results/m5_public_service_evidence_2026-05-17.json
```

Result:

```text
FAIL daemon heartbeat reports status=Offline
FAIL daemon serviceheartbeat reports status=NOT_SERVING
FAIL hosted heartbeat expected HTTP 200, got 503
FAIL hosted heartbeat reports endpoint unhealthy
FAIL hosted heartbeat error: endpoint unhealthy
```

Daemon heartbeat URL:

```text
https://1cf76242e956da6397b473a57f2ca4a9245568fe90b76910c1b18b4f.mainnet.haas.singularitynet.dev/heartbeat
```

Daemon heartbeat response summary:

```json
{
  "status": "Offline",
  "serviceheartbeat": "{\"serviceID\":\"zk_llm1\",\"status\":\"NOT_SERVING\"}",
  "daemonVersion": "v6.2.1",
  "blockchainEnabled": true,
  "blockchainNetwork": "main"
}
```

Hosted service heartbeat response:

```json
{
  "error": "endpoint unhealthy",
  "status": 503
}
```

## Request For SNET Support

Please explain why the Publisher UI reports both components as `UP` while:

- public daemon `/heartbeat` reports `status=Offline`
- daemon `serviceheartbeat` reports `status=NOT_SERVING`
- hosted-service orchestrator heartbeat returns HTTP 503 `endpoint unhealthy`

Please provide the Hosted Service serving/profiling logs and the current routing
configuration between:

```text
Hosted Service cb4c9cfb-bbf9-49b2-8357-d18926a7a5df
Daemon 2b4d4f3b-c043-4671-ad6a-697012038574
```

## Current Resubmission Status

Milestone 5 resubmission should remain blocked until public endpoint checks pass:

```bash
python3 scripts/m5_public_service_evidence.py \
  --output benchmarks/results/m5_public_service_evidence_2026-05-17.json
```

Required result:

```text
PASS wrote benchmarks/results/m5_public_service_evidence_2026-05-17.json
```
