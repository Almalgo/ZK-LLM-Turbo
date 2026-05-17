# Milestone 5 Resubmission Report

Status: blocked pending public SNET heartbeat correction.

This report is intentionally not ready for submission yet. The current public
endpoint checks still show the service as unavailable despite Publisher UI
showing both HaaS components as `UP`.

## Blocking Evidence

Current evidence artifact:

```text
benchmarks/results/m5_public_service_evidence_2026-05-17.json
```

Current result:

```text
status: fail
daemon_heartbeat_serving: false
hosted_heartbeat_healthy: false
```

Observed failures:

```text
daemon heartbeat reports status=Offline
daemon serviceheartbeat reports status=NOT_SERVING
hosted heartbeat expected HTTP 200, got 503
hosted heartbeat reports endpoint unhealthy
hosted heartbeat error: endpoint unhealthy
```

## Acceptance Criteria Before This Report Can Be Submitted

- Publisher screenshot shows Daemon `UP` and Hosted Service `UP`.
- Public daemon heartbeat does not report `Offline`.
- Public daemon heartbeat does not report `NOT_SERVING`.
- Hosted service heartbeat returns HTTP 200.
- At least one public functional service call succeeds.
- Reliability evidence shows `success_rate >= 0.95`.
- Recovery evidence shows `unrecovered_streaks = 0`.

## Verification Command

Run this after restarting/redeploying Hosted Service and Daemon:

```bash
python3 scripts/m5_public_service_evidence.py \
  --output benchmarks/results/m5_public_service_evidence_2026-05-17.json
```

Only replace this file with a submission-ready report after that command passes.
