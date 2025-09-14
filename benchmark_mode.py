from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

model.eval()

prompt = "Explain the benefits of homomorphic encryption."
inputs = tokenizer(prompt, return_tensors="pt")

start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32)
end = time.time()

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"Latency: {end - start:.2f} seconds")

