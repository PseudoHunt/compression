# === Setup ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from gptq import GPTQ
import math

# === CONFIG ===
MODEL_NAME = "facebook/opt-125m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
N_BATCHES = 5
SEQ_LEN = 32
NUM_BITS = 4
BLOCK_SIZE = 128
FIXED_T = 1000.0
LR = 0.001
NUM_ITERATIONS = 100

# === Load model and tokenizer ===
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# === Calibration Setup using TinyStories CSV ===
import pandas as pd

CSV_PATH = "validation.csv"        # Path to your TinyStories CSV
TEXT_COLUMN = "text"               # Column containing stories
N_CALIB_SAMPLES = 1000              # Number of samples to use

# Load and preprocess CSV
print("📖 Loading TinyStories from CSV...")
df = pd.read_csv(CSV_PATH)
assert TEXT_COLUMN in df.columns, f"'{TEXT_COLUMN}' column not found in CSV."
texts = df[TEXT_COLUMN].dropna().tolist()[:BATCH_SIZE * N_BATCHES]

# Tokenize
print("🔠 Tokenizing TinyStories for calibration...")
encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=SEQ_LEN,
    return_tensors="pt"
)
input_ids = encodings["input_ids"].to(DEVICE)
attention_mask = encodings["attention_mask"].to(DEVICE)
input_batches = input_ids.split(BATCH_SIZE)
mask_batches = attention_mask.split(BATCH_SIZE)

# === Define BlockwiseQuantizationOptim with GPTQ weight ===
class BlockwiseQuantizationOptim(nn.Module):
    def __init__(self, weight, block_size=128, num_bits=4, fixed_T=100.0,
                 gptq_scale=None, gptq_zero=None, gptq_g_idx=None):
        super().__init__()
        self.block_size = block_size
        self.num_bits = num_bits
        self.fixed_T = fixed_T
        self.original_shape = weight.shape
        self.num_levels = 2 ** num_bits

        padded_rows = math.ceil(weight.size(0) / block_size) * block_size
        padded_cols = math.ceil(weight.size(1) / block_size) * block_size
        self.padded_weight = torch.zeros((padded_rows, padded_cols), device=weight.device)
        self.padded_weight[:weight.size(0), :weight.size(1)] = weight

        self.blocks = []
        self.block_metadata = []
        for i in range(0, padded_rows, block_size):
            for j in range(0, padded_cols, block_size):
                block = self.padded_weight[i:i+block_size, j:j+block_size]
                self.blocks.append(block)
                self.block_metadata.append((i, j))

        self.w_min = nn.ParameterList()
        self.w_max = nn.ParameterList()

        for _, (i, j) in enumerate(self.block_metadata):
            if gptq_scale is not None and gptq_zero is not None and gptq_g_idx is not None:
                # Compute group indices for this block of columns
                col_start = j
                col_end = min(j + block_size, gptq_g_idx.shape[0])
                block_g_idx = gptq_g_idx[col_start:col_end]  # Shape: [block_cols]

                # Take the mean scale and zero for this block's group mapping
                scale_block = gptq_scale[0, block_g_idx].mean().detach()
                zero_block = gptq_zero[0, block_g_idx].mean().detach()

                # Derive min and max from scale/zero
                w_min = (-zero_block * scale_block)
                w_max = ((2 ** self.num_bits - 1 - zero_block) * scale_block)
            else:
                # Fallback to naive initialization
                block = self.padded_weight[i:i+block_size, j:j+block_size]
                w_min = block.min().detach()
                w_max = block.max().detach()

            self.w_min.append(nn.Parameter(w_min.view(1)))
            self.w_max.append(nn.Parameter(w_max.view(1)))


    def forward(self):
        eps = 1e-6
        q_blocks = []
        total_entropy = 0.0
        for idx, block in enumerate(self.blocks):
            w_min = self.w_min[idx].clamp(max=self.w_max[idx].item() - eps)
            w_max = self.w_max[idx].clamp(min=w_min.item() + eps)
            w_norm = (block - w_min) / (w_max - w_min + eps)
            q_levels = torch.linspace(0, 1, self.num_levels, device=block.device)
            dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
            soft_probs = torch.softmax(dists * self.fixed_T, dim=-1)
            w_q = (soft_probs * q_levels).sum(dim=-1)
            w_deq = w_q * (w_max - w_min) + w_min

            q_blocks.append(w_deq)

            bin_mass = soft_probs.sum(dim=0)
            bin_probs = bin_mass / (bin_mass.sum() + eps)
            entropy = -(bin_probs * (bin_probs + eps).log()).sum()
            total_entropy += entropy

        padded_out = torch.zeros_like(self.padded_weight)
        for idx, (i, j) in enumerate(self.block_metadata):
            padded_out[i:i+self.block_size, j:j+self.block_size] = q_blocks[idx]
        return padded_out[:self.original_shape[0], :self.original_shape[1]], total_entropy

    def export(self):
        eps = 1e-6
        q_blocks = []
        for idx, block in enumerate(self.blocks):
            w_min = self.w_min[idx].clamp(max=self.w_max[idx].item() - eps)
            w_max = self.w_max[idx].clamp(min=w_min.item() + eps)
            w_norm = (block - w_min) / (w_max - w_min + eps)
            q_levels = torch.linspace(0, 1, self.num_levels, device=block.device)
            dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
            q_idx = torch.argmax(dists, dim=-1).to(torch.int32)
            w_q = q_levels[q_idx]
            w_deq = w_q * (w_max - w_min) + w_min
            q_blocks.append(w_deq)

        padded_out = torch.zeros_like(self.padded_weight)
        for idx, (i, j) in enumerate(self.block_metadata):
            padded_out[i:i+self.block_size, j:j+self.block_size] = q_blocks[idx]
        return padded_out[:self.original_shape[0], :self.original_shape[1]].cpu()

# === Quantization Loop for all Linear Layers ===
safetensor_dict = {}
flag = 0
for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    # if flag == 4:
    #   break
    # flag += 1
    print(f"\n🔧 GPTQ + Blockwise Quantizing Layer: {name} | Shape: {module.weight.shape}")

    activation_batches = []
    def hook_fn(mod, inp, out):
        activation_batches.append(inp[0].detach())
    hook = module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, m in zip(input_batches, mask_batches):
            model(input_ids=x, attention_mask=m)
    hook.remove()

    if not activation_batches:
        continue

    gptq = GPTQ(module)
    for act in activation_batches:
        gptq.add_batch(act, module(act))
    scale, zero, g_idx = gptq.fasterquant(
        blocksize=BLOCK_SIZE,
        percdamp=0.01,
        group_size=128,
        actorder=True,
    )
    q_weight = module.weight.data.clone()

    # Init BlockwiseQuantizationOptim using GPTQ parameters
    quant_layer = BlockwiseQuantizationOptim(
        weight=module.weight.data.clone(),
        block_size=BLOCK_SIZE,
        num_bits=NUM_BITS,
        fixed_T=FIXED_T,
        gptq_scale=scale,
        gptq_zero=zero,
        gptq_g_idx=g_idx
    ).to(DEVICE)
    optimizer = torch.optim.Adam(quant_layer.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    #original_weight = module.weight.data.clone()

    for it in range(NUM_ITERATIONS):
        for act in activation_batches:
            optimizer.zero_grad()
            w_q, entropy = quant_layer()
            recon = F.linear(act.to(DEVICE), w_q)
            target = F.linear(act.to(DEVICE), q_weight)
            loss = mse_loss(recon, target) + mse_loss(q_weight, w_q)
            print(f"Iteration {it + 1}/{NUM_ITERATIONS}, Entropy: {entropy.item():.4f}, Loss: {loss.item():.8f}")
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_weight = quant_layer.export().to(module.weight.device)
        loss = mse_loss(q_weight, final_weight)
        print("weight diff",loss)
        module.weight.copy_(final_weight)
        safetensor_dict[name.replace(".", "_") + ".dequant"] = final_weight

# === Save Final Weights ===
#save_file(safetensor_dict, "quantized_blockwise_gptq.safetensors")
print("\n✅ Finished GPTQ-initialized blockwise quantization for all layers.")
