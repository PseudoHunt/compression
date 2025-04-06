# FG-Based LLM Quantization in Colab (with Activation-Aware Min/Max Calibration)

!pip install transformers accelerate -q

import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import numpy as np

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

def calibrate_activation_minmax(model, tokenizer, text, layer_name):
    activations = {}
    def capture(name):
        def hook(module, input, output):
            activations[name] = input[0].detach()
        return hook

    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(capture(name))

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

    return activations.get(layer_name, None)

def fg2_sensitivity(weight, loss_fn, epsilon=1e-3, num_samples=3):
    sensitivities = []
    for _ in range(num_samples):
        v = torch.randn_like(weight)
        v = v / v.norm()
        loss_plus = loss_fn(weight + epsilon * v)
        loss_minus = loss_fn(weight - epsilon * v)
        loss_center = loss_fn(weight)
        sensitivity = (loss_plus - 2 * loss_center + loss_minus) / (epsilon ** 2)
        sensitivities.append(sensitivity.item())
    return np.mean(sensitivities)

def quantize_llm_with_fg(model, tokenizer, sample_text, num_iter=20, lr=1e-2, epsilon=1e-3):
    model = copy.deepcopy(model).cpu()
    model.eval()
    layer_sensitivities = {}
    assigned_bits = {}

    inputs = tokenizer(sample_text, return_tensors="pt").to("cpu")
    labels = inputs["input_ids"]

    criterion = nn.CrossEntropyLoss()

    def compute_loss(m):
        with torch.no_grad():
            out = m(**inputs)
            return criterion(out.logits.view(-1, out.logits.size(-1)), labels.view(-1))

    # Estimate FG²-based sensitivities
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            baseline_loss = compute_loss(model)

            scores = []
            for _ in range(3):
                v = torch.randn_like(weight)
                v = v / v.norm()
                module.weight.data = weight + epsilon * v
                loss_plus = compute_loss(model)
                module.weight.data = weight - epsilon * v
                loss_minus = compute_loss(model)
                module.weight.data = weight
                sens = (loss_plus - 2 * baseline_loss + loss_minus) / (epsilon ** 2)
                scores.append(sens.item())
            layer_sensitivities[name] = np.mean(scores)

    # Assign bitwidths based on sensitivity
    sensitivities = list(layer_sensitivities.values())
    min_sens, max_sens = min(sensitivities), max(sensitivities)

    def scale_to_bits(s):
        norm = (s - min_sens) / (max_sens - min_sens + 1e-8)
        return int(np.clip(8 - 5 * norm, 3, 8))

    for name, sens in layer_sensitivities.items():
        bits = scale_to_bits(sens)
        assigned_bits[name] = bits
        print(f"Assigned {bits} bits to layer: {name} (sensitivity={sens:.6f})")

    # Quantize with FG-tuned w_min/w_max
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data.clone()
            original_weight = weight.clone()
            bitwidth = assigned_bits[name]

            # Activation-aware w_min/w_max init
            activation_input = calibrate_activation_minmax(model, tokenizer, sample_text, name)
            if activation_input is not None:
                input_range = activation_input.flatten().abs().mean()
                w_min = -input_range
                w_max = input_range
            else:
                w_min = weight.min().clone()
                w_max = weight.max().clone()

            # FG loss using output logits
            def fg_loss_fn(w_q, _):
                module.weight.data = w_q
                return compute_loss(model)

            w_min, w_max = forward_gradient_minmax_update(weight, w_min, w_max, fg_loss_fn,
                                                          epsilon=epsilon, lr=lr, num_iter=num_iter)
            module.weight.data = uniform_quantize(weight, w_min, w_max, num_bits=bitwidth)

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

print("\nRunning FG-based quantization with activation-aware calibration...")
quantized_model = quantize_llm_with_fg(model, tokenizer, sample_text, num_iter=20, lr=1e-2)
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
