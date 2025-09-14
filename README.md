# 🧠 ZK-LLM Split Inference: Phase 1 - Model Setup & Benchmarking

This project implements a privacy-preserving split-inference architecture using [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and CKKS-based homomorphic encryption.

This README covers **Phase 1** of the pipeline: setting up and benchmarking the base LLM in plaintext on a macOS environment.

---

## 📦 Setup

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
