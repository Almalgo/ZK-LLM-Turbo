# SNET HaaS Daemon Heartbeat Incident: zk_llm2

## Current symptom

The `zk_llm2` Daemon Only HaaS deployment can briefly appear online, then return
offline after the daemon heartbeat check.

Observed daemon logs:

```text
serviceHeartbeatURL: ""
Get "": unsupported protocol scheme ""
Service heartbeat unknown: Offline
```

## Affected deployment

```text
Organization: almalgo_labs
Service ID: zk_llm2
Daemon ID: de4cc165-8a62-48ed-bfb1-4bb62bf99d51
Runtime: Coolify
Backend base URL: https://zkllm.almalgo.com
Backend heartbeat URL: https://zkllm.almalgo.com/heartbeat
```

## Backend verification

The backend is not the failing component. It should return HTTP 200:

```bash
curl -i https://zkllm.almalgo.com/heartbeat
curl -i -X POST https://zkllm.almalgo.com/health \
  -H 'content-type: application/json' \
  -d '{"op":"health"}'
```

Expected response body includes:

```json
{
  "serviceID": "zk_llm2"
}
```

## Root cause

The live HaaS daemon config is internally inconsistent:

```text
service_heartbeat_type = http
heartbeat_endpoint = ""
```

For SNET daemon HTTP heartbeat mode, the daemon calls the configured
`heartbeat_endpoint`. If the endpoint is empty while the type is `http`, it tries
to call an empty URL and fails with:

```text
Get "": unsupported protocol scheme ""
```

## Docs and source evidence

SNET HaaS Daemon Only UI documentation exposes only:

```text
Service Endpoint
Authorization
```

The visible UI does not expose:

```text
heartbeat_endpoint
service_heartbeat_type
```

The SNET daemon heartbeat implementation supports these relevant modes:

```text
http / https: call heartbeat_endpoint
none / empty / tcp: ping service_endpoint
```

Therefore `service_heartbeat_type=http` with `heartbeat_endpoint=""` is the bad
combination. It must be changed server-side by HaaS or avoided with the fallback.

## Preferred fix

Ask SNET support to set the live HaaS daemon config to:

```text
service_endpoint = https://zkllm.almalgo.com
service_heartbeat_type = http
heartbeat_endpoint = https://zkllm.almalgo.com/heartbeat
```

## Fallback fix

If HaaS Daemon Only cannot set `heartbeat_endpoint`, ask SNET support to set:

```text
service_endpoint = https://zkllm.almalgo.com
service_heartbeat_type = none
heartbeat_endpoint = ""
```

This avoids `GET ""` and lets the daemon ping the base service endpoint instead.

## Exact support request

```text
The Daemon Only HaaS config for my service is generated inconsistently.

Organization: almalgo_labs
Service ID: zk_llm2
Daemon ID: de4cc165-8a62-48ed-bfb1-4bb62bf99d51
Service endpoint: https://zkllm.almalgo.com

Current daemon logs show:
service_heartbeat_type = http
serviceHeartbeatURL = ""
Get "": unsupported protocol scheme ""
Service heartbeat unknown: Offline

Please update the live HaaS daemon config to:

service_endpoint = https://zkllm.almalgo.com
service_heartbeat_type = http
heartbeat_endpoint = https://zkllm.almalgo.com/heartbeat

If HaaS Daemon Only cannot set heartbeat_endpoint, please set:
service_heartbeat_type = none

The backend heartbeat endpoint is healthy:
GET https://zkllm.almalgo.com/heartbeat -> 200
Response includes serviceID zk_llm2.

The Publisher UI only exposes Service Endpoint and Authorization for Daemon Only,
so I cannot set heartbeat_endpoint from the UI.

Please redeploy the daemon using v6.2.2 if available.
```

## What not to do

Do not set the visible Publisher Service Endpoint to:

```text
https://zkllm.almalgo.com/heartbeat
```

That can make heartbeat checks pass, but it breaks real HTTP passthrough calls,
which require the base URL:

```text
POST /health
POST /session
POST /layer
```

Do not use Publisher Authorization fields to try to set daemon config keys.
Authorization is passed as service credentials to the backend; it does not set
daemon heartbeat configuration.

## Acceptance criteria

The fix is complete when daemon logs no longer show:

```text
serviceHeartbeatURL: ""
Get "": unsupported protocol scheme ""
```

Daemon heartbeat should report `Online` and wrap:

```json
{"serviceID":"zk_llm2","status":"SERVING"}
```

The marketplace should remain online across at least one periodic heartbeat
cycle.
