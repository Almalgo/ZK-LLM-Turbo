# 1. Abstract

This report documents the design, implementation, and evaluation of Milestone 4 (MVP & Tests v0.1) of the ZK-LLM Turbo project — the first working prototype of privacy-preserving split inference using homomorphic encryption.

The system now enables:
- True split inference: server performs linear operations (matrix multiplications) on encrypted data
- Client handles all non-linear operations (RMSNorm, SiLU, softmax, attention) in plaintext
- Privacy-preserving protocol: server receives only a public CKKS context (no secret key) and cannot decrypt any data
- Configurable encrypted layers (default 1) with remaining layers processed in plaintext via PyTorch
- End-to-end token generation using TinyLlama 1.1B with greedy decoding
- **Batched HTTP requests**: all tokens in a generation step are sent in a single request per round (4 requests per step instead of 4 x seq_len)
- **KV caching**: after the initial prompt, only the single new token is processed through all layers — encrypted layer caches K/V in NumPy, plaintext layers use PyTorch's DynamicCache
- **Interactive CLI**: users can send arbitrary prompts, with optional `--logs` and `--stats` flags

This milestone transitions the project from a stub architecture (server decrypted and returned dummy data) to real encrypted computation with practical performance optimizations.

# 2. Project Objective

Build a working prototype of privacy-preserving LLM inference where:
- The server performs linear algebra on encrypted data without ever seeing plaintext
- The client retains full control over non-linear operations and the secret key
- The architecture supports configurable encrypted layers for performance/privacy tradeoffs

Milestone 4 implements the complete split-inference protocol for TinyLlama's decoder layers and validates correctness against plaintext PyTorch inference.

# 3. Development Environment

- Hardware: WSL2 on Windows (Linux 6.6.87.2-microsoft-standard-WSL2)
- Operating System: Ubuntu (Python 3.12.3)
- Language: Python 3.12 within a virtual environment (`.venv/`)
- Server runtime: FastAPI on Uvicorn with lifespan model loading
- Client runtime: Python with NumPy for non-linear ops
- Encryption library: TenSEAL (CKKS scheme)
- Model: TinyLlama 1.1B Chat v1.0 (HuggingFace)

# 4. Architecture: Split Inference Protocol

## 4.1 Core Principle

LLM transformer layers consist of two types of operations:
- **Linear operations** (matrix multiplications): can be performed homomorphically on encrypted data
- **Non-linear operations** (RMSNorm, SiLU, softmax, attention): cannot be performed homomorphically and must be done in plaintext

The split-inference protocol separates these: the server handles linear ops on encrypted data, and the client handles non-linear ops after decrypting intermediate results.

## 4.2 Protocol: 4 Round-Trips Per Layer

Each TinyLlama decoder layer requires 4 client-server round-trips. All tokens in a step are batched into a single HTTP request per round:

```
Round 1: Q/K/V Projections
  Client: RMSNorm(X) → encrypt all tokens → send batch
  Server: For each token: Enc(Q)=Enc(X)@W_q, Enc(K)=Enc(X)@W_k, Enc(V)=Enc(X)@W_v
  Server → Client: [Enc(Q0),Enc(K0),Enc(V0), Enc(Q1),Enc(K1),Enc(V1), ...]

Round 2: O Projection
  Client: decrypt Q,K,V → RoPE → attention(softmax(QK^T/sqrt(d))·V) → encrypt
  Server: For each token: Enc(o_out) = Enc(attn) @ W_o
  Server → Client: [Enc(o0), Enc(o1), ...]

Round 3: FFN Gate + Up Projections
  Client: decrypt → residual + RMSNorm → encrypt
  Server: For each token: Enc(gate)=Enc(X)@W_gate, Enc(up)=Enc(X)@W_up
  Server → Client: [gate_parts..., up_parts...] per token

Round 4: FFN Down Projection
  Client: decrypt → SiLU(gate)*up → encrypt chunks
  Server: For each token: Enc(down) = sum(Enc(part_i) @ W_down_i)
  Server → Client: [Enc(down0), Enc(down1), ...]
  Client: decrypt → residual → next layer input
```

## 4.3 KV Caching

After the initial prompt is processed, subsequent token generation uses KV caching to avoid redundant computation:

**Encrypted layer cache** (client-side NumPy):
- After Round 1, the client stores the rotated K and V vectors for all processed tokens
- On subsequent steps, only the new token's Q/K/V are computed via the server (1 encrypt + 1 HTTP request)
- Attention is computed using the new Q against the full cached K/V

**Plaintext layer cache** (PyTorch DynamicCache):
- Each plaintext decoder layer stores its KV pairs in a shared `DynamicCache` object
- On subsequent steps, only the new token's hidden state is passed through each layer
- PyTorch's SDPA handles causal masking automatically (q_len=1 attends to all past)

**Impact**: For a 6-token prompt generating 5 tokens, total tokens processed through all layers drops from 40 (6+7+8+9+10) to 10 (6+1+1+1+1) — a 4x reduction.

## 4.4 Dimension Splitting

CKKS with `poly_modulus_degree=8192` provides 4096 slots per ciphertext. TinyLlama's FFN intermediate dimension is 5632, which exceeds this limit. The solution:
- **Split output** (gate_proj, up_proj: 2048 → 5632): split weight matrix columns into chunks of ≤4096, producing 2 ciphertexts
- **Split input** (down_proj: 5632 → 2048): split input vector and weight matrix rows, multiply each chunk, sum the encrypted results

## 4.5 Session Management

The client creates a CKKS context with a secret key, then serializes it **without the secret key** and sends it to the server. The server stores this public context per session and uses it for all HE operations. The server can compute on encrypted data but cannot decrypt any ciphertexts.

# 5. Implementation Details

## 5.1 Bug Fixes (Step 0)

- Renamed `requirements.text` → `requirements.txt` and resolved merge conflicts
- Removed hardcoded prompt override in `client/client.py:37`
- Fixed `server/tests/test_api_endpoint.py` import (`server.server.app`)
- Enabled `use_relin_keys: true` in client config (required for HE matmul)
- Added Python artifacts to `.gitignore` (`__pycache__/`, `*.pyc`, `.env`)
- Added `server/tests/__init__.py` for pytest discovery

## 5.2 Embedding Extraction Fix (Step 1)

**Problem**: The original `extract_embeddings()` ran a full forward pass through all 22 layers using `AutoModel`, returning `last_hidden_state`. This was architecturally wrong — we need the initial embedding before any layers process it.

**Fix**: Changed to use `model.model.embed_tokens(input_ids)` only, which returns the raw token embeddings (seq_len x 2048) without any layer processing. Switched to `AutoModelForCausalLM` for access to `lm_head`. Added model caching to avoid reloading 1.1B parameters on every call.

## 5.3 Client-Side Non-Linear Operations (Step 6)

Implemented in `client/inference/nonlinear_ops.py`:

| Operation | Description | Validation |
|-----------|-------------|------------|
| `rms_norm(x, weight, eps)` | Matches PyTorch `LlamaRMSNorm` | Tested against PyTorch reference |
| `silu(x)` | x * sigma(x), matches `torch.nn.functional.silu` | Tested against PyTorch reference |
| `softmax(x, axis)` | Numerically stable (subtract max) | Tested: sums to 1, handles large values |
| `compute_attention(q, k, v)` | GQA with 32 query / 4 KV heads, causal mask | Shape tests, single-token correctness |
| `apply_rotary_embeddings(q, k)` | RoPE positional encoding | Integrated in layer protocol |
| `apply_rotary_embeddings_at_positions(q, k, positions)` | RoPE with explicit position indices for KV cache | Used in incremental generation |
| `compute_attention_cached(q, k, v)` | Attention supporting q_len != kv_len for KV cache | Initial pass (causal) + incremental (no mask) |

## 5.4 Server-Side HE Operations (Step 4)

Implemented in `server/inference/he_ops.py`:

| Function | Input -> Output | Use Case |
|----------|----------------|----------|
| `he_matmul(enc, W)` | Enc(D_in) -> Enc(D_out) | Q, K, V, O projections |
| `he_matmul_split_output(enc, W)` | Enc(D_in) -> [Enc(chunk)...] | gate_proj, up_proj (2048->5632) |
| `he_matmul_split_input(encs, W, sizes)` | [Enc(chunk)...] -> Enc(D_out) | down_proj (5632->2048) |

All use TenSEAL's `enc_vector.mm(weight_matrix)` which computes the encrypted vector-matrix product.

## 5.5 Server Endpoint Design (Steps 3, 5)

Two endpoints:

**`POST /api/session`** — Receives base64-encoded public CKKS context, stores per session, returns `session_id`.

**`POST /api/layer`** — Accepts batched encrypted vectors:
```json
{
  "session_id": "...",
  "layer_idx": 0,
  "operation": "qkv",
  "encrypted_vectors_b64": ["token0_b64", "token1_b64", "..."],
  "chunk_sizes": [4096, 1536]
}
```
Operations: `qkv`, `o_proj`, `ffn_gate_up`, `ffn_down`. All operations loop over input vectors to support batched processing. For `ffn_down`, the batch size is inferred from `len(vectors) // len(chunk_sizes)`. Returns encrypted results without ever accessing plaintext.

The model is loaded once at server startup via FastAPI lifespan. Layer weights are extracted, transposed (Linear stores `(out, in)`, matmul needs `(in, out)`), and cached per layer.

## 5.6 Client Orchestrator (Step 8)

The `generate()` function implements the full inference pipeline:

1. Tokenize prompt
2. Extract initial embeddings (`embed_tokens` only)
3. Create CKKS context and send public context to server
4. For each token to generate:
   a. Extract embeddings (all tokens on step 0, only new token on subsequent steps)
   b. For layers 0..N-1: process encrypted via `EncryptedLayerProtocol` with KV cache
   c. For layers N..21: process plaintext via PyTorch with DynamicCache
   d. Apply final RMSNorm + lm_head -> logits
   e. Greedy decode -> append token -> repeat

The number of encrypted layers is configurable (default 1).

## 5.7 Performance Optimizations

### Batched HTTP Requests

Previously, each token required a separate HTTP request per round — for a 6-token prompt, that was 24 HTTP requests just for Round 1. Now all tokens are batched into a single request per round.

| Metric | Before | After |
|--------|--------|-------|
| HTTP requests per step (6 tokens) | 24 | 4 |
| HTTP requests per step (1 token, cached) | 4 | 4 |
| Total for 6-token prompt + 5 generated | ~160 | ~20 |

### KV Caching

| Component | Cache Type | Effect |
|-----------|-----------|--------|
| Encrypted layers | NumPy arrays (K, V after RoPE) | Only new token through 4 HE round-trips |
| Plaintext layers | PyTorch DynamicCache | Only new token through 21 decoder layers |

The DynamicCache is pre-filled with empty placeholder tensors for encrypted layer indices so that plaintext layer indices align correctly.

### Combined Impact

For a 6-token prompt generating 5 tokens:
- **HE matmuls**: reduced from ~400 to ~100 (4x fewer)
- **Plaintext layer passes**: reduced from 21 x 40 = 840 to 21 x 10 = 210 (4x fewer)
- **HTTP round-trips**: reduced from ~160 to ~20 (8x fewer)

## 5.8 CLI Interface

The client now supports arbitrary prompts and configurable behavior:

```bash
python -m client.client [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | *(interactive)* | Input text for inference |
| `--num-tokens` | 5 | Number of tokens to generate |
| `--num-encrypted-layers` | 1 | Layers processed via encrypted server inference |
| `--logs` | off | Enable verbose JSON logging (silent by default) |
| `--stats` | off | Print per-phase timing breakdown table |

When `--stats` is enabled, the client prints a table after generation:

```
+-------------------------+----------+--------+
| Phase                   | Time (s) |      % |
+-------------------------+----------+--------+
| Model loading           |    12.34 |  18.3% |
| Session setup           |     1.23 |   1.8% |
| Embedding extraction    |     0.15 |   0.2% |
| Encrypted layers        |    45.67 |  67.8% |
| Plaintext layers        |     7.89 |  11.7% |
| Token decode            |     0.12 |   0.2% |
+-------------------------+----------+--------+
| Total                   |    67.40 |        |
+-------------------------+----------+--------+
```

# 6. TinyLlama Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 2048 |
| FFN intermediate dimension | 5632 |
| Number of decoder layers | 22 |
| Number of query heads | 32 |
| Number of KV heads (GQA) | 4 |
| Head dimension | 64 |
| Vocabulary size | 32000 |
| RMSNorm epsilon | 1e-5 |
| Activation function | SiLU |
| Position encoding | RoPE |

# 7. CKKS Encryption Parameters

| Parameter | Value |
|-----------|-------|
| Scheme | CKKS |
| `poly_modulus_degree` | 8192 |
| `coeff_mod_bit_sizes` | [60, 40, 40, 60] |
| `global_scale` | 2^40 |
| Available slots | 4096 |
| Galois keys | Enabled |
| Relinearization keys | Enabled |

These parameters provide a balance between security, precision, and computational depth. The 4096 slot limit is the primary architectural constraint, requiring dimension splitting for FFN operations.

# 8. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `requirements.txt` | Renamed + fixed | Was `requirements.text` with merge conflicts |
| `.gitignore` | Updated | Added Python artifacts |
| `pytest.ini` | Updated | Added server/tests, slow marker |
| `client/config/client_config.yaml` | Updated | Enabled relin keys, added inference config |
| `client/config/endpoints.yaml` | Updated | Added session/layer endpoints |
| `client/model/embedding_extractor.py` | Rewritten | embed_tokens only, caching, model components |
| `client/encryption/ckks_context.py` | Updated | Added `serialize_public_context()` |
| `client/client.py` | Rewritten | Full generate() pipeline with KV cache, argparse CLI, --logs/--stats |
| `client/inference/__init__.py` | New | Package init |
| `client/inference/nonlinear_ops.py` | Updated | Added `apply_rotary_embeddings_at_positions`, `compute_attention_cached` |
| `client/inference/layer_protocol.py` | Rewritten | Batched requests, KV caching, position-aware RoPE |
| `server/server.py` | Updated | Lifespan model loading, new routers |
| `server/handlers/inference_handler.py` | Rewritten | `/api/layer` with batched token processing |
| `server/handlers/session_handler.py` | New | `/api/session` endpoint |
| `server/model/__init__.py` | New | Package init |
| `server/model/weight_manager.py` | New | Weight extraction and caching |
| `server/inference/__init__.py` | New | Package init |
| `server/inference/he_ops.py` | New | HE matmul operations |
| `server/requirements.txt` | Updated | Added torch, transformers |
| `server/tests/__init__.py` | New | Package init |
| `server/tests/test_api_endpoint.py` | Fixed | Corrected import path |
| `server/tests/test_he_ops.py` | New | HE matmul tests at multiple scales |
| `client/tests/test_embeddings.py` | Updated | Fixed assertions for embed_tokens |
| `client/tests/test_nonlinear_ops.py` | New | Nonlinear ops vs PyTorch |
| `client/tests/test_public_context.py` | New | Public context security tests |
| `client/tests/test_e2e_accuracy.py` | New | Encrypted vs plaintext accuracy |
| `notebooks/milestone4_demo.ipynb` | Updated | Added KV caching and batching demos |
| `README.md` | Updated | Local run guide, CLI options, KV caching notes |

# 9. Testing Strategy

## 9.1 Non-Linear Operations (`client/tests/test_nonlinear_ops.py`)

- RMSNorm: matches PyTorch `LlamaRMSNorm` within float32 tolerance (rtol=1e-5)
- SiLU: matches `torch.nn.functional.silu` within 1e-6
- Softmax: sums to 1, numerically stable with large inputs, matches PyTorch
- Attention: correct output shape, single-token edge case

## 9.2 Public Context Security (`client/tests/test_public_context.py`)

- Verifies serialized public context does not contain secret key
- Verifies server can perform HE computation (plaintext-ciphertext multiply)
- Verifies server can perform HE matmul with public context
- Verifies client can decrypt results computed by server

## 9.3 End-to-End Accuracy (`client/tests/test_e2e_accuracy.py`)

- Encrypt-decrypt roundtrip within CKKS tolerance (~0.001)
- Encrypted matmul matches plaintext at 64->32 dims (atol=0.05)
- Encrypted matmul at full 2048->256 dims (atol=0.1)
- RMSNorm determinism
- SiLU correctness

## 9.4 HE Operations (`server/tests/test_he_ops.py`)

- Small-dimension matmul (8->4) within 0.01 tolerance
- Hidden-dimension matmul (2048->256) within 0.1 tolerance
- Full hidden-to-hidden matmul (2048->2048) within 0.5 tolerance
- Split-output matmul for dims > 4096 slots
- Split-input matmul with chunk reconstruction

## 9.5 Demonstration Notebook (`notebooks/milestone4_demo.ipynb`)

Seven sections:
1. CKKS public context proof (server cannot decrypt)
2. HE matrix multiplication at multiple scales with error analysis
3. Non-linear operations comparison vs PyTorch
4. Timing breakdown: encrypt, serialize, matmul, decrypt per token
5. Accuracy comparison: encrypted vs plaintext Q-projection with cosine similarity
6. KV caching: demonstrate cached vs uncached attention computation
7. Batching: show reduced HTTP request count

# 10. Challenges & Tradeoffs

## 10.1 CKKS Slot Limitation

The 4096 slot limit (from `poly_modulus_degree=8192`) requires splitting FFN vectors (dim 5632) across 2 ciphertexts. This doubles the number of encryptions, serializations, and network transfers for FFN operations.

**Tradeoff**: Increasing `poly_modulus_degree` to 16384 would provide 8192 slots (enough for 5632) but would significantly increase ciphertext sizes and computation time.

## 10.2 Per-Token Processing

Each token in the sequence must be encrypted and processed individually, as CKKS operates on fixed-length vectors. With batched requests and KV caching, the cost is now O(4 x num_encrypted_layers) HTTP requests per generated token (after the initial prompt), rather than O(4 x seq_len x num_encrypted_layers).

## 10.3 Accumulated CKKS Error

CKKS is an approximate scheme — each operation introduces small numerical errors. For a single matmul at 2048 dimensions, the error is typically < 0.1. Across multiple chained operations (Q projection -> attention -> O projection -> FFN), errors may accumulate. Using only 1 encrypted layer by default mitigates this.

## 10.4 Weight Transposition

PyTorch `nn.Linear` stores weights as `(out_features, in_features)`. HE matmul requires `(in_features, out_features)` for `x @ W`. All weights must be transposed before use. This is handled transparently in `weight_manager.py`.

## 10.5 Embedding Extraction Bug

The original implementation used `AutoModel` with a full forward pass (`last_hidden_state`), which meant the "embeddings" already had all 22 layers applied. This made split inference meaningless. Fixed by using `embed_tokens` only.

## 10.6 DynamicCache Index Alignment

When using PyTorch's `DynamicCache` for plaintext layers while skipping encrypted layers, the cache's internal list must have placeholder entries for the skipped layer indices. Without pre-filling, `DynamicCache.update()` appends at the wrong list index, causing an `IndexError`. Fixed by pre-filling with empty `(1, num_kv_heads, 0, head_dim)` tensors for each encrypted layer.

# 11. Security Model

## 11.1 What the Server Sees

- Public CKKS context (public key, galois keys, relin keys — no secret key)
- Encrypted ciphertexts (cannot be decrypted without secret key)
- Model weights (the server loads TinyLlama locally)

## 11.2 What the Server Cannot Do

- Decrypt any ciphertext
- Recover plaintext embeddings, attention patterns, or intermediate states
- Determine the input prompt or generated tokens

## 11.3 What the Client Controls

- Secret key (never leaves the client)
- All non-linear operations (executed in plaintext on client)
- Final token selection (logits -> greedy decode)

## 11.4 Trust Assumptions

- The CKKS scheme is semantically secure under the RLWE assumption
- The server honestly performs the requested linear operations (verified via ZK proofs in future milestones)
- The network transport is authenticated (Bearer token)

# 12. Next Steps

- ~~Batched token processing: reduce HTTP round-trips~~ **Done** (batched requests)
- ~~Performance benchmarking: per-phase timing~~ **Done** (`--stats` flag)
- **Multi-layer encrypted inference**: Test with 2+ encrypted layers and measure accuracy degradation
- **Zero-knowledge proofs**: Introduce verifiable computation proofs for server operations
- **Payload compression**: Reduce ciphertext sizes for faster network transfer
- **Ciphertext depth management**: Monitor and manage multiplicative depth across chained operations
- **Streaming output**: Display tokens as they are generated rather than waiting for full completion
- **GPU acceleration**: Offload plaintext layer inference to GPU for faster processing

# 13. Conclusion

Milestone 4 delivers the first working prototype of privacy-preserving split inference for TinyLlama. The server performs matrix multiplications on encrypted data using only a public CKKS context, while the client handles all non-linear operations in plaintext. The architecture supports configurable encrypted layers, allowing tradeoffs between privacy and performance.

Key achievements:
- Real homomorphic computation replaces the dummy stub from Milestone 3
- Public context sharing ensures the server cannot decrypt any data
- Dimension splitting handles FFN operations that exceed CKKS slot limits
- Batched HTTP requests reduce network round-trips by up to 8x
- KV caching reduces redundant computation by up to 4x for multi-token generation
- Interactive CLI with `--logs` (verbose debugging) and `--stats` (timing breakdown) flags
- Comprehensive test suite validates correctness against PyTorch references
- Interactive notebook demonstrates all building blocks with timing and accuracy analysis

This milestone validates the feasibility of encrypted transformer inference and establishes the foundation for zero-knowledge correctness proofs in future milestones.
