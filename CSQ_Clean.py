import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from scipy.stats import entropy

# Load model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16)

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


# quantize_all_linear_inplace.py
import math
import numpy as np
from scipy.stats import entropy

import torch
import torch.nn as nn


# ---------- Utilities ----------
def calculate_entropy(int_weights: np.ndarray) -> float:
    values, counts = np.unique(int_weights.flatten(), return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs, base=2)

def calculate_mse(original: torch.Tensor, dequantized: torch.Tensor) -> float:
    return ((original - dequantized) ** 2).mean().item()


# ---------- Group-wise quant/dequant (RTN) ----------
def _groupwise_minmax_quant_dequant(x: torch.Tensor, n_bits=4, group_size=128):
    """
    Group-wise (flattened) min-max RTN quantization + dequantization.
    Returns (q:int32, dq:float32) with same shape as x.
    """
    qmin, qmax = 0, (1 << n_bits) - 1
    shape = x.shape
    flat = x.view(-1)
    num_groups = (flat.numel() + group_size - 1) // group_size

    q = torch.empty_like(flat, dtype=torch.int32)
    dq = torch.empty_like(flat, dtype=torch.float32)

    for g in range(num_groups):
        s = g * group_size
        e = min((g + 1) * group_size, flat.numel())
        grp = flat[s:e]
        min_val, max_val = grp.min(), grp.max()

        scale = (max_val - min_val) / (qmax - qmin + 1e-6)
        scale = scale.clamp_min(1e-8)

        qi = torch.clamp(((grp - min_val) / scale).round(), qmin, qmax)
        q[s:e] = qi.to(torch.int32)
        dq[s:e] = qi * scale + min_val

    return q.view(shape), dq.view(shape)

# ---------- Group-wise UNIFORM quant/dequant ----------
def _groupwise_uniform_quant_dequant(
    x: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 128,
    symmetric: bool = True,
):
    """
    Group-wise UNIFORM quantization + dequantization over the flattened tensor.
      - symmetric=True: q in [-qmax, qmax], scale = max_abs / qmax
      - symmetric=False: asymmetric min-max (affine) with zero_point

    Returns:
      q  : int32 codes with same shape as x
      dq : float32 dequantized weights with same shape as x
    """
    shape = x.shape
    flat = x.view(-1)
    num_groups = (flat.numel() + group_size - 1) // group_size

    if symmetric:
        qmax = (1 << (n_bits - 1)) - 1  # e.g., 7 for 4-bit
        qmin = -qmax
    else:
        qmax = (1 << n_bits) - 1        # e.g., 15 for 4-bit
        qmin = 0

    q = torch.empty_like(flat, dtype=torch.int32)
    dq = torch.empty_like(flat, dtype=torch.float32)

    eps = 1e-8
    for g in range(num_groups):
        s = g * group_size
        e = min((g + 1) * group_size, flat.numel())
        grp = flat[s:e]

        if symmetric:
            max_abs = grp.abs().max()
            scale = (max_abs / (qmax + 1e-6)).clamp_min(eps)
            qi = torch.clamp((grp / scale).round(), qmin, qmax)
            q[s:e] = qi.to(torch.int32)
            dq[s:e] = qi * scale
        else:
            # Asymmetric min-max (affine) – similar to RTN but explicit zero-point
            min_val, max_val = grp.min(), grp.max()
            # Avoid degenerate range
            scale = ((max_val - min_val) / (qmax - qmin + 1e-6)).clamp_min(eps)
            zero_point = torch.clamp((qmin - min_val / scale).round(), qmin, qmax).to(torch.int32)

            qi = torch.clamp((grp / scale + zero_point).round(), qmin, qmax).to(torch.int32)
            q[s:e] = qi
            dq[s:e] = (qi.to(torch.float32) - zero_point.to(torch.float32)) * scale

    return q.view(shape), dq.view(shape)




# ---------- Cosine Snapping Quantization (CSQ) ----------
def _groupwise_csq_quant_dequant(x: torch.Tensor, n_bits=4, group_size=128, delta=8, lambd=0.8):
    """
    Cosine Snapping Quantization: per-sample scale = scale * alpha_i,
    where alpha_i = 1 + λ cos(2π e_i / δ), e_i ≈ rounded bin index at half-resolution.

    We quantize with (scale * alpha_i) but dequantize with the base 'scale'
    to avoid storing alphas (matches your earlier approximation).
    Returns (q:int32, dq:float32, alpha:float32).
    """
    qmin, qmax = 0, (1 << n_bits) - 1
    shape = x.shape
    flat = x.view(-1)
    num_groups = (flat.numel() + group_size - 1) // group_size

    q = torch.empty_like(flat, dtype=torch.int32)
    dq = torch.empty_like(flat, dtype=torch.float32)
    alpha_all = torch.empty_like(flat, dtype=torch.float32)
    two_pi = 2 * math.pi

    for g in range(num_groups):
        s = g * group_size
        e = min((g + 1) * group_size, flat.numel())
        grp = flat[s:e]
        min_val, max_val = grp.min(), grp.max()

        scale = ((max_val - min_val) / (qmax - qmin + 1e-6)).clamp_min(1e-8)
        # # half-resolution index (like your e_i formula)
        # e_i = ((grp - min_val) / (2 * scale)).round()

        # alpha = 1.0 + lambd * torch.cos((two_pi * e_i) / delta)

        q3 = ((grp - min_val) / (scale * 2)).round()
        q4_target = 2 * q3
        alpha = np.divide(grp.detach().numpy()-min_val.detach().numpy(), q4_target.detach().numpy() * scale.detach().numpy(), out=np.ones_like(grp.detach().numpy()), where=q4_target.detach().numpy() != 0)
        alpha = torch.from_numpy(alpha)



        qi = torch.clamp(((grp - min_val) / (scale * alpha)).round(), qmin, qmax)

        q[s:e] = qi.to(torch.int32)
        alpha_all[s:e] = alpha.to(torch.float32)
        # storage-free dequant (no alpha used on decode)
        dq[s:e] = qi * scale + min_val

    return q.view(shape), dq.view(shape), alpha_all.view(shape)


# ---------- Top-level API ----------
def quantize_model_inplace(
    model: nn.Module,
    method: str = "csq4",           # 'csq4', 'rtn4', or 'rtn3'
    group_size: int = 128,
    delta: int = 8,
    lambd: float = 0.8,
    skip_last_lm_head: bool = True,
    skip_name_substrings = ("lm_head", "embed_out", "output_projection", "classifier", "score"),
):
    """
    Walks all nn.Linear layers, quantizes+DEquantizes, writes DEquantized weights back in-place.
    Skips the final LM head layer by default.

    Returns:
      entropies: dict[name] -> entropy(bits)
      mses:      dict[name] -> MSE
    """
    entropies, mses = {}, {}

    # Collect linear modules with names for stable ordering
    linear_modules = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]

    # Try to detect an LM head by common names; else use the last Linear as a conservative fallback
    head_candidates = [n for n, _ in linear_modules if any(s in n for s in skip_name_substrings)]
    fallback_head = linear_modules[-1][0] if linear_modules else None
    final_head_name = head_candidates[-1] if head_candidates else fallback_head

    with torch.no_grad():
        for name, mod in linear_modules:
            if skip_last_lm_head and (any(s in name for s in skip_name_substrings) or name == final_head_name):
                # Skip LM head / final projection
                continue

            W = mod.weight.data
            orig_dtype = W.dtype
            Wf = W.float().contiguous()

            if method == "rtn4":
                q, dq = _groupwise_minmax_quant_dequant(Wf, n_bits=4, group_size=group_size)
            elif method == "rtn3":
                q, dq = _groupwise_minmax_quant_dequant(Wf, n_bits=3, group_size=group_size)
            elif method == "csq4":
                q, dq, _ = _groupwise_csq_quant_dequant(Wf, n_bits=4, group_size=group_size, delta=delta, lambd=lambd)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Write back DEquantized weights in-place
            mod.weight.copy_(dq.to(orig_dtype))

            # Metrics
            entropies[name] = calculate_entropy(q.cpu().numpy())
            mses[name] = calculate_mse(Wf, dq)
            print(f"  {name:40s}  entropy={entropies[name]:.3f}  mse={mses[name]:.6f}")

    return entropies, mses


# ---------- Example usage ----------
if __name__ == "__main__":
    torch.manual_seed(0)

    # If you have a real model, just replace 'model = DummyLLM()' with it.
    #model = DummyLLM()
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16)

    # Choose a method: 'csq4' (cosine snapping, 4-bit), 'rtn4' (plain 4-bit), or 'rtn3' (plain 3-bit)
    method = "rtn3"
    ent, mse = quantize_model_inplace(
        model,
        method=method,
        group_size=128,
        delta=8,
        lambd=0.8,
        skip_last_lm_head=True,  # don't touch the final LM head
        # You can extend this list if your head layer has a different name:
        skip_name_substrings=("lm_head", "embed_out", "output_projection", "classifier", "score"),
    )

    # Quick glance
    first3 = list(ent.items())[:3]
    print(f"[{method}] processed layers: {len(ent)} (lm_head skipped)")
    for k, v in first3:
        print(f"  {k:40s}  entropy={v:.3f}  mse={mse[k]:.6f}")


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# Assume you have your quantized model object as `model`
# (weights have already been replaced with dequantized versions)
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare a sample input
prompt = "The future of artificial intelligence is not"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,      # control length of output
        do_sample=True,         # enable randomness
        temperature=0.7,        # creativity
        top_p=0.9               # nucleus sampling
    )

# Decode
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Prepare a sample input
prompt = "I like travelling"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,      # control length of output
        do_sample=True,         # enable randomness
        temperature=0.7,        # creativity
        top_p=0.9               # nucleus sampling
    )

# Decode
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
