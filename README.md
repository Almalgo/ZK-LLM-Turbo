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

## Running Locally

Running split inference requires **two terminals** — one for the server and one for the client.

### 1. Start the Server

```bash
cd /path/to/ZK-LLM-Turbo
uvicorn server.server:app --reload --host 0.0.0.0 --port 8000
```

The server loads TinyLlama 1.1B at startup and exposes two endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /api/session` | Establish encrypted session (receives public CKKS context) |
| `POST /api/layer` | Perform homomorphic operations on encrypted vectors |

### 2. Run the Client

Once the server is ready, run the client from a second terminal:

```bash
cd /path/to/ZK-LLM-Turbo
python -m client.client --prompt "The capital of France is"
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--prompt` | *(interactive)* | Input text for inference. If omitted, the client prompts you interactively. |
| `--num-tokens` | `5` | Number of tokens to generate |
| `--num-encrypted-layers` | `1` | How many initial layers to process via encrypted server inference |
| `--logs` | off | Enable verbose JSON logging output (silent by default) |
| `--stats` | off | Print a timing breakdown table after generation |

**Examples:**

```bash
# Interactive mode — the client will ask for your prompt
python -m client.client

# Specify everything via CLI
python -m client.client --prompt "Once upon a time" --num-tokens 10 --num-encrypted-layers 2

# Enable verbose logging for debugging
python -m client.client --prompt "Hello world" --logs

# Show timing breakdown
python -m client.client --prompt "Hello world" --stats
```

The client will:

1. Load the TinyLlama tokenizer and model
2. Create a CKKS encryption context and send the **public** context to the server
3. Run split inference — encrypted layers on the server (4 round-trips each) + plaintext layers locally
4. Print per-token progress with timing, then the full generated text

KV caching is used automatically: the first token processes the full prompt, then each subsequent token only processes the single new token through all layers. This significantly reduces inference time for multi-token generation.

### Optional: Expose Server via HTTPS

```bash
ngrok http 8000
```

Then update `client/config/endpoints.yaml` with the ngrok HTTPS URL.

---

## Running Tests

```bash
pytest -q              # all quick tests
pytest -m slow         # slow tests (downloads model)
pytest -v              # verbose output
```

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

## License

Private research prototype.

---

## 🙌 Acknowledgements

* HuggingFace TinyLlama
* TenSEAL / Zama
* FastAPI community
* ZK-LLMS project

---
