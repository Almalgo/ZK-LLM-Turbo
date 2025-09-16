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

### 4. Model's layers

```bash
<class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>
model <class 'transformers.models.llama.modeling_llama.LlamaModel'>
model.embed_tokens <class 'torch.nn.modules.sparse.Embedding'>
model.layers <class 'torch.nn.modules.container.ModuleList'>
model.layers.0 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.0.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.0.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.0.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.0.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.0.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.0.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.1 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.1.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.1.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.1.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.1.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.1.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.1.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.2 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.2.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.2.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.2.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.2.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.2.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.2.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.3 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.3.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.3.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.3.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.3.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.3.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.3.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.4 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.4.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.4.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.4.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.4.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.4.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.4.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.5 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.5.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.5.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.5.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.5.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.5.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.5.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.6 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.6.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.6.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.6.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.6.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.6.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.6.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.7 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.7.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.7.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.7.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.7.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.7.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.7.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.8 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.8.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.8.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.8.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.8.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.8.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.8.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.9 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.9.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.9.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.9.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.9.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.9.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.9.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.10 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.10.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.10.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.10.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.10.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.10.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.10.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.11 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.11.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.11.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.11.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.11.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.11.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.11.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.12 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.12.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.12.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.12.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.12.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.12.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.12.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.13 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.13.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.13.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.13.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.13.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.13.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.13.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.14 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.14.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.14.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.14.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.14.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.14.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.14.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.15 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.15.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.15.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.15.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.15.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.15.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.15.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.16 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.16.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.16.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.16.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.16.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.16.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.16.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.17 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.17.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.17.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.17.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.17.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.17.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.17.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.18 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.18.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.18.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.18.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.18.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.18.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.18.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.19 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.19.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.19.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.19.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.19.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.19.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.19.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.20 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.20.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.20.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.20.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.20.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.20.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.20.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.21 <class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>
model.layers.21.self_attn <class 'transformers.models.llama.modeling_llama.LlamaAttention'>
model.layers.21.self_attn.q_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.self_attn.k_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.self_attn.v_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.self_attn.o_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.mlp <class 'transformers.models.llama.modeling_llama.LlamaMLP'>
model.layers.21.mlp.gate_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.mlp.up_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.mlp.down_proj <class 'torch.nn.modules.linear.Linear'>
model.layers.21.mlp.act_fn <class 'torch.nn.modules.activation.SiLU'>
model.layers.21.input_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.layers.21.post_attention_layernorm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.norm <class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>
model.rotary_emb <class 'transformers.models.llama.modeling_llama.LlamaRotaryEmbedding'>
lm_head <class 'torch.nn.modules.linear.Linear'>
```
## Phase2: CLIENT-SIDE DEVELOPMENT
### 1. 