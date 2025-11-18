# ZK-LLM Turbo — Homomorphic Encryption Client/Server Split Inference

### *Privacy-Preserving LLM Embeddings Using CKKS + TinyLlama*

---

## Overview

**ZK-LLM Turbo** implements a full client–server encrypted inference workflow using:

* **TinyLlama 1.1B** tokenization + embedding extraction
* **CKKS homomorphic encryption (TenSEAL)**
* **Encrypted transport** to a FastAPI server
* **Encrypted partial inference** (mocked in Phase 2)
* **Encrypted results returned to the client**

The core objective is to demonstrate how to perform the *entire embedding pipeline on the client*, encrypt the output, and transmit it securely without exposing plaintext embeddings.

This repository contains:

* Full **client implementation**
* Full **server implementation**
* **Structured JSON logging** utilities
* **Complete test suite** (unit + integration)
* A **public Jupyter notebook** demonstrating the encryption and transport pipeline

---

## Deliverables

### Tokenization

Using TinyLlama tokenizer from HuggingFace.

### Embedding Extraction

Extract final hidden-state embeddings from the TinyLlama model.

### CKKS Encryption via TenSEAL

Parameters:

* poly modulus degree: **8192**
* coeff mod bit sizes: **[60, 40, 40, 60]**
* global scale: **2^40**

One encrypted vector per token is produced, serialized with base64.

### JSON Payload Format

```
{
  "encrypted_embeddings": ["b64...", "b64..."],
  "metadata": {
    "cid": "uuid",
    "embedding_shape": [seq_len, dim],
    "ckks": {...}
  }
}
```

### Server-side Handling

* Deserialize → decrypt (optional diagnostic)
* Perform mocked inference
* Return encrypted result to client

### Structured Logging

Each log entry is JSON:

```
{
  "timestamp": "...",
  "level": "INFO",
  "message": "Payload prepared",
  "cid": "...",
  "payload_bytes": 4456887
}
```

### Full Test Suite

* Unit tests: tokenizer, embeddings, CKKS, encryption, payload
* Integration: client end-to-end (mocked)
* Server API endpoint tests

---

## Public Notebook

A notebook demonstrating:

* Synthetic TinyLlama-shaped embeddings
* CKKS encryption/decryption
* Payload building
* Simulated server handling

File: `notebooks/phase2_hybrid_demo.ipynb`

Runs fully offline.

---

## Installation

### 1. Create environment

```
python3 -m venv split-inference-env
source split-inference-env/bin/activate
```

### Install dependencies

```
pip install -r requirements.txt
```

---

## ▶Running the Client

```
cd client
python client.py
```

Performs:

* Tokenization
* Embedding extraction
* CKKS encryption
* Payload build
* HTTPS POST
* Decrypts result
* Logs everything

---

## Running the Server

```
cd server
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Optional: expose via HTTPS

```
ngrok http 8000
```

Update client’s `endpoints.yaml` to match HTTPS URL.

---

## Running Tests

```
pytest -q
```

All public tests run offline.

---

## Logging Format

Structured logs include:

* timestamp
* cid (correlation ID)
* latency
* payload size
* status codes
* errors

---

## Phase 3 Preview

* Encrypted linear layer evaluation
* Galois rotations for attention
* Zero-knowledge validity proofs
* GPU-friendly CKKS

---

## License

Private research prototype.

---

## 🙌 Acknowledgements

* HuggingFace TinyLlama
* TenSEAL / Zama
* FastAPI community
* ZK-LLMS project

---
