# Re-run with a simple dummy weight matrix since `transformers` is unavailable in this environment

import torch
import numpy as np
import math
from scipy.stats import entropy

# Utilities
def calculate_entropy(int_weights):
    values, counts = np.unique(int_weights.flatten(), return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def calculate_mse(original, dequantized):
    return ((original - dequantized) ** 2).mean().item()

def quantize_tensor(tensor, n_bits=4, group_size=128):
    qmin = 0
    qmax = (1 << n_bits) - 1
    shape = tensor.shape
    tensor = tensor.view(-1)
    num_groups = (tensor.numel() + group_size - 1) // group_size
    quantized = torch.empty_like(tensor, dtype=torch.int32)
    dequantized = torch.empty_like(tensor)
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, tensor.numel())
        group = tensor[start:end]
        min_val = group.min()
        max_val = group.max()
        scale = (max_val - min_val) / (qmax - qmin + 1e-6)
        scale = max(scale, 1e-8)
        q = torch.clamp(((group - min_val) / scale).round(), qmin, qmax)
        quantized[start:end] = q.int()
        dequantized[start:end] = q * scale + min_val
    return quantized.view(shape), dequantized.view(shape)

def quantize_tensor_with_ei_alpha(tensor, n_bits=4, group_size=128, delta=8, lambd=1.0):
    qmin = 0
    qmax = (1 << n_bits) - 1
    shape = tensor.shape
    tensor = tensor.view(-1)
    num_groups = (tensor.numel() + group_size - 1) // group_size
    quantized = torch.empty_like(tensor, dtype=torch.int32)
    alpha_all = torch.empty_like(tensor)
    dequantized = torch.empty_like(tensor)
    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, tensor.numel())
        group = tensor[start:end]
        min_val = group.min()
        max_val = group.max()
        R = max_val - min_val
        scale = R / (qmax - qmin + 1e-6)
        scale = max(scale, 1e-8)
        e_i = ((group - min_val) / (2 * scale)).round()
        alpha = 1 + lambd * torch.cos(2 * math.pi * e_i / delta)

        q3 = ((group - min_val) / (scale * 2)).round()
        q4_target = 2 * q3
        alpha = np.divide(group.detach().numpy()-min_val.detach().numpy(), q4_target.detach().numpy() * scale.detach().numpy(), out=np.ones_like(group.detach().numpy()), where=q4_target.detach().numpy() != 0)
        alpha = torch.from_numpy(alpha)
        q = ((group - min_val) / (scale * alpha)).round()
        #q = ((group * alpha - min_val) / scale).round()
        q = torch.clamp(q, qmin, qmax)
        quantized[start:end] = q.int()
        alpha_all[start:end] = alpha
        dequantized[start:end] = (q * scale + min_val) #* alpha
    return quantized.view(shape), dequantized.view(shape), alpha_all.view(shape)

# Dummy input weight matrixf
#W = torch.randn(256, 256)
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from scipy.stats import entropy

# Load model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16)


layer = model.model.decoder.layers[0].self_attn.q_proj
W = layer.weight.float().cpu()

# Entropy and MSE collection
entropies = {}
mses = {}

# RTN 4-bit
q4, dq4 = quantize_tensor(W, n_bits=4, group_size=128)
entropies["grpwise-RTN-4bit"] = calculate_entropy(q4.numpy())
mses["grpwise-RTN-4bit"] = calculate_mse(W, dq4)

# RTN 3-bit
q3, dq3 = quantize_tensor(W, n_bits=3, group_size=128)
entropies["grpwise-RTN-3bit"] = calculate_entropy(q3.numpy())
mses["grpwise-RTN-3bit"] = calculate_mse(W, dq3)

# CSQ 4-bit
q_ei, dq_ei, _ = quantize_tensor_with_ei_alpha(W, n_bits=4, group_size=128, delta=8, lambd=0.8)
entropies["CSQ"] = calculate_entropy(q_ei.numpy())
mses["CSQ"] = calculate_mse(W, dq_ei)

(entropies, mses)
