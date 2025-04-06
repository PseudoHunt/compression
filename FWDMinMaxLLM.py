# FG-Based LLM Quantization in Colab (for OPT-125M / GPT2 / etc.)

!pip install transformers accelerate -q

import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# --- Quantization Core ---
def uniform_quantize(weight, w_min, w_max, num_bits=8):
    qmin, qmax = 0, 2 ** num_bits - 1
    scale = (w_max - w_min) / (qmax - qmin)
    scale = scale.clamp(min=1e-8)
    zero_point = qmin - w_min / scale
    q = ((weight / scale) + zero_point).round().clamp(qmin, qmax)
    return (q - zero_point) * scale

def forward_gradient_minmax_update(weight, w_min, w_max, loss_fn, epsilon=1e-3, lr=1e-2, num_iter=20):
    for _ in range(num_iter):
        v = torch.randn(2, device=weight.device)
        v = v / v.norm()
        w_min_pert = w_min + epsilon * v[0]
        w_max_pert = w_max + epsilon * v[1]
        q_pert = uniform_quantize(weight, w_min_pert, w_max_pert)
        q_orig = uniform_quantize(weight, w_min, w_max)
        loss_pert = loss_fn(q_pert, weight)
        loss_orig = loss_fn(q_orig, weight)
        grad_est = (loss_pert - loss_orig) / epsilon * v
        w_min = w_min - lr * grad_est[0]
        w_max = w_max - lr * grad_est[1]
    return w_min, w_max

def quantize_llm_with_fg(model, num_bits=8, num_iter=20, lr=1e-2):
    model = copy.deepcopy(model).cpu()
    loss_fn = nn.MSELoss()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                weight = module.weight.data
                w_min = weight.min().clone().detach()
                w_max = weight.max().clone().detach()
                w_min, w_max = forward_gradient_minmax_update(weight, w_min, w_max, loss_fn, lr=lr, num_iter=num_iter)
                module.weight.data = uniform_quantize(weight, w_min, w_max, num_bits=num_bits)
    return model

# --- Perplexity Evaluation ---
def compute_perplexity(model, tokenizer, text):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

# --- Load and Quantize a Hugging Face Model ---
model_name = "facebook/opt-125m"  # Or "gpt2", "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cuda")

# Perplexity before quantization
sample_text = "The quick brown fox jumps over the lazy dog."
ppl_before = compute_perplexity(model, tokenizer, sample_text)
print(f"Perplexity before quantization: {ppl_before:.2f}")

print("\nRunning FG-based quantization...")
quantized_model = quantize_llm_with_fg(model, num_bits=8, num_iter=20, lr=1e-2)
quantized_model = quantized_model.to("cuda")

# Perplexity after quantization
ppl_after = compute_perplexity(quantized_model, tokenizer, sample_text)
print(f"Perplexity after quantization: {ppl_after:.2f}")

# --- Test Inference ---
quantized_model.eval()
tokenizer.pad_token = tokenizer.eos_token
prompt = "Once upon a time in Bangalore,"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = quantized_model.generate(**inputs, max_length=50)
    print("\nQuantized Model Output:\n", tokenizer.decode(output[0]))
