# Milestone 6: MVP & Tests Report

Date: 2026-04-26

## Goal

Deliver Milestone 6 with:
1. Codebase (privately shared through GitHub)
2. Public Jupyter notebook for public tests
3. Final report

## Summary

This milestone focused on hardening the ZK-LLM-Turbo codebase for MVP release with improved security, reliability, observability, and test coverage.

### Security Improvements

| Change | File | Description |
|--------|------|-------------|
| Auth token configurable | `client/config/endpoints.yaml` | Removed hardcoded token, now requires env var |
| Auth enabled by default | `server/security.py` | Auth enabled when token is set |
| Session cleanup | `client/client.py` | Sessions explicitly deleted on client exit |

### Authentication Compatibility (Milestone 6 Follow-up)

- Added server auth token precedence while preserving legacy env vars:
  - `ZKLLM_SERVER_AUTH_TOKEN`
  - `ZKLLM_API_TOKEN`
  - `AUTH_TOKEN`
- Kept explicit auth opt-out via `ZKLLM_REQUIRE_API_TOKEN=false`.
- Added validation for this path in `server/tests/test_security.py`.

### Reliability Improvements

| Change | File | Description |
|--------|------|-------------|
| Session cleanup on exit | `client/client.py` | finally block deletes session |
| Error logging | `client/client.py` | Added error logging on network failures |
| Configurable timeouts | `client/inference/layer_protocol.py` | WebSocket timeouts now configurable |

### Test Coverage

Added 11 new test files and one security follow-up test update:

- `client/tests/test_encrypt_embeddings.py`
- `client/tests/test_encryption_utils.py`
- `client/tests/test_session_expiry.py`
- `client/tests/test_websocket_reconnection.py`
- `server/tests/test_request_limits.py`
- `server/tests/test_security.py`
- `server/tests/test_decryption_utils.py`
- `client/tests/test_client_payload_validation.py`
- `server/tests/test_server_payload_validation.py`
- `client/tests/test_e2e_session_lifecycle.py`
- `client/tests/test_e2e_network_errors.py`

Total test count: 152 tests

### Public Deliverables

1. **Public Jupyter Notebook**: `notebooks/public_mvp_demo.ipynb`
   - Demonstrates privacy guarantee (server cannot decrypt)
   - Shows HE matrix multiplication
   - Compares non-linear ops with PyTorch
   - Timing breakdown

2. **This report**: Documents all changes and improvements

## Architecture Summary

| Component | Details |
|-----------|---------|
| Model | TinyLlama 1.1B (22 layers) |
| Encryption | CKKS via TenSEAL |
| Polynomial degree | 8192 (4096 slots) |
| Protocol | 4 round-trips per encrypted layer |
| Transport | HTTP + WebSocket |

## Test Command

```bash
pytest -m "not slow"
```

Validation at report close:

- `pytest -m "not slow"` -> 152 passed, 3 deselected
- `pytest -m "slow"` -> 3 passed, 152 deselected

## Known Limitations

1. **Multi-token packing**: Deferred (TenSEAL API incompatibility)
2. **Selective Galois keys**: Deferred (no documented API)
3. **OpenFHE backend**: Experimental (32x slower than TenSEAL)
4. **GPU acceleration**: No usable GPU path yet

## Deferred from Previous Milestones

- T1.3: Multi-token packing (concatenation-style incompatible with CKKSVector.mm())
- T1.4: Selective Galois keys (no safe API found)
- T3.1: OpenFHE migration (no_go - performance)
- T3.3: GPU acceleration (no_go - no usable path)
- T3.4: Full polynomial model (server-side polynomials missing)
- T3.5: Non-interactive protocol (blocked by T3.3, T3.4)

## Files Modified

### Security
- `client/config/endpoints.yaml`
- `server/security.py`
- `client/client.py`

### Reliability
- `client/client.py` (session cleanup)
- `client/inference/layer_protocol.py` (configurable timeouts)

### Tests (11 new files)
- All listed in Test Coverage section

### Documentation
- `notebooks/public_mvp_demo.ipynb`

### Milestone 6 Commits (latest)

- `8fe9243` — Update security and session hardening with tests
- `5465f9e` — Refresh milestone 6 test count metadata
- `6d42d3b` — Add server auth token alias and update milestone count

## Performance Characteristics

| Operation | Typical Latency |
|------------|-----------------|
| Encrypt (2048-dim) | ~50ms |
| Serialize | ~10ms |
| HE matmul 2048→2048 | ~100ms |
| Decrypt | ~5ms |
| Ciphertext size | ~50KB |

## Privacy Guarantees

1. **Server receives public context only** - cannot decrypt any ciphertexts
2. **Session isolation** - each session has separate CKKS context
3. **No plaintext exposure** - all linear ops computed homomorphically
4. **Client-side non-linear ops** - RMSNorm, SiLU, Softmax run in plaintext

## Version

v0.2 - MVP Release
