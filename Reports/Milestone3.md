# 1. Abstract

This report summarizes the design, implementation, and evaluation of Phase 2 of the ZK-LLM Turbo project — a prototype architecture for performing client-side encrypted embedding generation as a foundational component of a future Zero-Knowledge LLM inference system.

The system enables:
- Tokenization and embedding extraction using TinyLlama.
- Encryption of token embeddings using CKKS homomorphic encryption via TenSEAL.
- Transmission of encrypted embeddings to a remote server.
- Server-side handling and encrypted response generation.
- Full roundtrip measurement and logging

This phase establishes the foundation for future fully homomorphic inference and verifiable correctness proofs.

# 2. Project Objective

The core objective of the project is:

Build a prototype for a fully Zero-Knowledge LLM inference system, where the server can perform inference on encrypted data without ever accessing plaintext embeddings.

Phase 2 focuses specifically on the client-side modules, ensuring the entire input pipeline (token → embedding → encryption → payload → send) operates correctly, reproducibly, and efficiently.

3. Development Environment

- Hardware: MacBook Pro M3 Pro, 32 GB RAM
- Operating System: macOS
- Language: Python 3.13 within a virtual environment
- Server runtime: FastAPI on Uvicorn
- Client runtime: Pure Python, synchronous execution
- Encryption library: TenSEAL (CKKS)

Both client and server were developed and tested locally.

# 4. Model & Tokenization Decisions

### 4.1 Model Choice — TinyLlama 1.1B

TinyLlama was selected for:
- Size: Easily runnable locally for embedding extraction
- Speed: Fast enough for interactive prototyping
- Compatibility: Standard HuggingFace architecture

### 4.2 Embedding Dimensions

For Phase 2 tests, we used:
- Token count: 10 tokens
- Embedding dimension: 2048 (final hidden layer size)

This is consistent with TinyLlama’s architecture and offers a realistic payload size for encrypted inference.

# 5. Homomorphic Encryption Design

### 5.1 Why TenSEAL?

TenSEAL was chosen because:

- PySEAL is deprecated & unmaintained
- Microsoft SEAL has no modern Python bindings
- TenSEAL is actively maintained
- TenSEAL directly supports CKKS vectors, the format needed for encrypted embeddings

### 5.2 CKKS Parameters

Phase 2 uses testing parameters, not final production values:
- poly_modulus_degree: 8192
- coeff_mod_bit_sizes: [60, 40, 40, 60]
- global_scale: 2^40

These parameters produce:
- Encrypted vectors ~300–500 KB each
- Payloads ~4–5 MB for 10 encrypted embeddings

More parameter sweeps will be conducted before the beta release.

# 6. System Architecture

### 6.1 Client Pipeline

tokenize → extract embeddings → CKKS encrypt → serialize → build payload → send

Detailed steps:
- Load TinyLlama tokenizer
- Tokenize user prompt
- Run TinyLlama to generate embeddings
- Encrypt each token embedding row with CKKS
- Serialize to base64
- Construct payload with metadata
- Send over HTTPS (ngrok for MVP)

### 6.2 Server Pipeline

receive → JSON decode → deserialize CKKS → decrypt → generate dummy encrypted response

Detailed steps:
- Receive POST request
- Validate + log payload
- Deserialize encrypted vectors
- CKKS-decrypt first vector for debugging
- Encrypt dummy response
- Return encrypted result

This architecture will support plugging in encrypted linear layers in Phase 3.

# 7. Performance Benchmarks

Based on logs and observations during development, the following estimates are representative:

### 7.1 Latency (Estimated)
Stage:	Estimated Time
Tokenization:	5–10 ms
Embedding extraction:	15–30 ms
Encryption (10×2048 CKKS vectors):	40–60 ms
Payload build:	10–20 ms
HTTP (local + ngrok tunnel):	3000–5200 ms
Server processing:	30–60 ms
Client decryption of server result:	2–5 ms
Total Roundtrip ~3.5–5.5 seconds (dominated by HTTPS tunnel + large payload size)

This is typical for CKKS with 10×2048 vectors.

### 7.2 Payload Size

Average size: ~4.4–4.6 MB

Encrypted CKKS vector for a 2048-dim embedding is ~350–500 KB

10 vectors → ~4–5 MB total payload

# 8. Challenges & Tradeoffs

### 8.1 Large CKKS Payloads

CKKS encryption expands plaintext vectors significantly (10–20×). This leads to large network payloads, ngrok introduced additional latency overhead

### 8.2 TenSEAL Import Stability

TenSEAL requires specific build environments.

Some Python 3.13 compatibility issues surfaced.

### 8.3 Tokenizer Download Issues

HuggingFace downloads do not work well inside mocked tests

Required mocking functions to simulate TinyLlama tokenization

### 8.4 FastAPI Request Models

Distinguishing between Request and Pydantic models caused errors

Proper parsing required reading raw body manually

### 8.5 Path/Import Complexity

Needed consistent project-root handling

Tests failed initially due to missing __init__.py

# 9. Testing Strategy

A full test suite was implemented:

### 9.1 Unit Tests

- CKKS context creation

- encryption/decryption

- embedding extraction

- payload builder

- tokenizer (mocked to avoid HF network calls)

### 9.2 Integration Tests

Client → mocked server response

Tests avoided real network calls via requests_mock

### 9.3 Server Tests

FastAPI /api/infer response structure

Deserialization and minimal pipeline validation

### 9.4 Lessons Learned

Must mock HF tokenizers in CI

Must avoid real network calls

Must simulate embeddings to keep tests stable

# 10. Public Notebook

A reproducible notebook (phase2_hybrid_demo.ipynb) was created.

It demonstrates:

- Synthetic TinyLlama-shaped embeddings

- Real CKKS encryption via TenSEAL

- Payload construction identical to production client

- A simulated server endpoint

- Decryption of server response

- Timing and analysis

This notebook is ideal for milestone evaluation.

11. Next Steps (Phase 3 Preview)

The next milestone includes launching the full beta of ZK-LLM Turbo, including:

- Integration of encrypted linear layers
- CKKS-friendly matrix multiplication
- Galois rotations for attention blocks
- Partial encrypted transformer inference
- Accuracy evaluation under CKKS approximation
- Introducing Zero-Knowledge correctness proofs
- Community testing & external evaluation
- Optimization of payload batching and network protocol

The goal is to assemble the first end-to-end prototype of encrypted, verifiable LLM inference.

12. Conclusion

Phase 2 successfully demonstrates the foundation required for fully encrypted and eventually zero-knowledge LLM inference:

- Client-side embedding generation
- CKKS encryption
- Encrypted transport
- Server-side encrypted response handling
- Logging, testing, and reproducibility infrastructure

This milestone validates the feasibility of encrypted embedding pipelines and prepares the project for encrypted transformer operations in Phase 3.