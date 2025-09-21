# 🧠 ZK-LLM Split Inference: Phase 1 - Model Setup & Benchmarking

This project implements a privacy-preserving split-inference architecture using [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and CKKS-based homomorphic encryption.

This README covers **Phase 1** of the pipeline: setting up and benchmarking the base LLM in plaintext on a macOS environment.

---

## Phase1: Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ZK-LLM-Turbo.git
cd ZK-LLM-Turbo
```
### 2. Create virtual environment

```bash
python3.10 -m venv split-inference-env
source split-inference-env/bin/activate
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📆 Phase 2: Client-Side Development

This phase implements the **client-side logic** of our privacy-preserving LLM system, where the user prompt is **embedded and encrypted locally** using **CKKS homomorphic encryption (via TenSEAL)** before being sent securely to the server.

### 🚀 Overview

> In this phase, we:

1. Tokenize the user prompt using TinyLlama’s tokenizer.
2. Generate token embeddings using TinyLlama’s input embedding layer.
3. Encrypt the embeddings using TenSEAL with CKKS.
4. Transmit the encrypted vectors to the server via HTTPS.



### ✅ Prerequisites

* Python 3.10+
* Virtual environment recommended (`venv` or `pyenv`)
* `transformers` for model and tokenizer
* `tenseal` for CKKS encryption
* `requests` for HTTP communication


### 📅 Install Dependencies

```bash
pip install transformers tenseal requests
```

### 🔐 Notes

* Both client and server must use **identical CKKS parameters**.
* `poly_modulus_degree = 8192` supports up to 4096 slots (floats per ciphertext).
* Only the embedding layer is run locally — the rest of the model runs homomorphically (or approximated) on the server.
* For production: use **HTTPS**, authentication tokens, and **TLS pinning** for secure transport.



### 📛 References

* [TinyLlama on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
* [TenSEAL GitHub](https://github.com/OpenMined/TenSEAL)
* [CKKS Explained](https://eprint.iacr.org/2017/565.pdf) (original paper)
