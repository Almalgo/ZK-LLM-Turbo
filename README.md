# 🧠 ZK-LLM Split Inference: Phase 1 - Model Setup & Benchmarking

This project implements a privacy-preserving split-inference architecture using [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and CKKS-based homomorphic encryption.

---

## Setup

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

<<<<<<< HEAD
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
=======
---


## **Phase 1: Model Setup & Benchmark Report**

**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
**Environment:** macOS (Apple Silicon M3 Pro, 32 GB RAM)
**Backend:** PyTorch 2.1 (CPU)
**Date:** 2025-11-05



### 1. Benchmark Results

| Metric                  | Description                                                                        | Result           |
| ----------------------- | ---------------------------------------------------------------------------------- | ---------------- |
| **Latency (prompt)**    | Time to generate completion for: “Explain the benefits of homomorphic encryption.” | **0.58 seconds** |
| **Memory footprint**    | Resident memory after model load (`psutil`).                                       | **5953.2 MB RAM** |
| **Baseline perplexity** | On Wikitext-2 (1 % subset). Lower = better.                                        | **PPL = 6.13**   |

*(All results measured on CPU; expect faster inference on GPU or Metal.)*


### 2. Model Architecture Summary

**Total parameters:** ≈ 1.1 B
**Architecture:** LLaMA-style transformer decoder with rotary positional embeddings.

| Layer type     | Count | Description                                         | Homomorphic Potential         |
| -------------- | ----- | --------------------------------------------------- | ----------------------------- |
| `nn.Linear`    | 160 + | Weight matrices for attention QKV + MLP projections | ✅ (Encryptable under CKKS)    |
| `nn.Embedding` | 2     | Token + position embeddings                         | ✅ (Can be encrypted input)    |
| `nn.GELU`      | 32    | Non-linear MLP activation                           | ⚠️ (Approximation needed)     |
| `Softmax`      | 32    | Attention normalization                             | ⚠️ (Needs polynomial approx.) |
| `LayerNorm`    | 64    | Normalization layers                                | ⚠️ (Not directly HE-friendly) |

A text export of the model layers (`model_layers.txt`) has been generated via:

```python
for name, module in model.named_modules():
    print(name, type(module))
```


### 📈 3. Observations

* **Model loaded successfully on macOS CPU**, verifying plaintext inference path.
* **Latency is acceptable** for testing (0.3 s response, ~15 tokens/s).
* **Memory usage** fits within typical laptop constraints.
* **Perplexity** provides a baseline for future encrypted inference comparison.
* **Linear layers dominate computation**, making them the primary target for homomorphic evaluation in Phase 2.


---



>>>>>>> f38776e (phase 1 & 2 changes)
