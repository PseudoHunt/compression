# === Setup ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
#from gptq import GPTQ
import math

# === CONFIG ===
MODEL_NAME = "facebook/opt-125m"
#MODEL_NAME = "databricks/dolly-v2-3b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
N_BATCHES = 5
SEQ_LEN = 32
NUM_BITS = 3
BLOCK_SIZE = 128
FIXED_T = 1000.0
LR = 0.001
NUM_ITERATIONS = 1

# === Load model and tokenizer ===
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",                # Automatically split layers across available GPUs/CPU
    torch_dtype="float32",               # Use float16 where possible
    low_cpu_mem_usage=True            # Efficient weight loading
)
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
def get_power_bins(a=0.5, num_bits=4, device="cpu"):
    q_levels = 2 ** num_bits
    lin = torch.linspace(0, 1, q_levels, device=device)
    scaled = (lin ** (1 / a)) * 0.5
    bins = 0.5 + torch.cat([-scaled.flip(0), scaled[1:]])
    return bins
# === Define BlockwiseQuantizationOptim with GPTQ weight ===
class BlockwiseQuantizationOptim(nn.Module):
    def __init__(self, weight, block_size=128, num_bits=4, fixed_T=100.0, use_blockwise=True, a=0.5):
        super().__init__()
        self.block_size = block_size
        self.num_bits = num_bits
        self.fixed_T = fixed_T
        self.a = a
        self.original_shape = weight.shape
        self.use_blockwise = use_blockwise
        self.num_levels = 2 ** num_bits

        if use_blockwise:
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
            self.learnable_bins = nn.ParameterList()
            for block in self.blocks:
                w_min, w_max = block.min().detach(), block.max().detach()
                pad = 0.05 * (w_max - w_min)
                self.w_min.append(nn.Parameter((w_min - pad).view(1)))
                self.w_max.append(nn.Parameter((w_max + pad).view(1)))
                init_bins = self._get_power_bins().detach()
                self.learnable_bins.append(nn.Parameter(init_bins.clone()))
        else:
            self.blocks = [weight]
            self.block_metadata = [(0, 0)]
            w_min, w_max = weight.min().detach(), weight.max().detach()
            pad = 0.05 * (w_max - w_min)
            self.w_min = nn.Parameter((w_min - pad).view(1))
            self.w_max = nn.Parameter((w_max + pad).view(1))
            init_bins = self._get_power_bins().detach()
            self.learnable_bins = nn.Parameter(init_bins.clone())

    def _get_power_bins(self):
        lin = torch.linspace(0, 1, 2 ** self.num_bits)
        scaled = (lin ** (1 / self.a)) * 0.5
        bins = 0.5 + torch.cat([-scaled.flip(0), scaled[1:]])
        return bins

    def forward(self):
        eps = 1e-6
        q_blocks = []
        total_entropy = 0.0

        if not self.use_blockwise:
            blocks = [self.blocks[0]]
            w_mins = [self.w_min]
            w_maxs = [self.w_max]
            bin_lists = [self.learnable_bins]
        else:
            blocks = self.blocks
            w_mins = self.w_min
            w_maxs = self.w_max
            bin_lists = self.learnable_bins

        for idx, block in enumerate(blocks):
            w_min = w_mins[idx].clamp(max=w_maxs[idx].item() - eps)
            w_max = w_maxs[idx].clamp(min=w_min.item() + eps)
            w_norm = (block - w_min) / (w_max - w_min + eps)

            bins = bin_lists[idx].to(block.device)
            dists = -torch.abs(w_norm.unsqueeze(-1) - bins)
            soft_probs = torch.softmax(dists * self.fixed_T, dim=-1)
            w_q = (soft_probs * bins).sum(dim=-1)
            w_deq = w_q * (w_max - w_min) + w_min
            q_blocks.append(w_deq)

            # Entropy penalty
            bin_mass = soft_probs.sum(dim=0)
            bin_probs = bin_mass / (bin_mass.sum() + eps)
            entropy = -(bin_probs * (bin_probs + eps).log()).sum()
            total_entropy += entropy

        if self.use_blockwise:
            padded_out = torch.zeros_like(self.padded_weight)
            for idx, (i, j) in enumerate(self.block_metadata):
                padded_out[i:i+self.block_size, j:j+self.block_size] = q_blocks[idx]
            return padded_out[:self.original_shape[0], :self.original_shape[1]], total_entropy
        else:
            return q_blocks[0], total_entropy

    def export(self):
        eps = 1e-6
        q_blocks = []

        if not self.use_blockwise:
            blocks = [self.blocks[0]]
            w_mins = [self.w_min]
            w_maxs = [self.w_max]
            bin_lists = [self.learnable_bins]
        else:
            blocks = self.blocks
            w_mins = self.w_min
            w_maxs = self.w_max
            bin_lists = self.learnable_bins

        for idx, block in enumerate(blocks):
            w_min = w_mins[idx].clamp(max=w_maxs[idx].item() - eps)
            w_max = w_maxs[idx].clamp(min=w_min.item() + eps)
            w_norm = (block - w_min) / (w_max - w_min + eps)

            bins = bin_lists[idx].to(block.device)
            dists = -torch.abs(w_norm.unsqueeze(-1) - bins)
            q_idx = torch.argmax(dists, dim=-1).to(torch.int32)
            w_q = bins[q_idx]
            w_deq = w_q * (w_max - w_min) + w_min
            q_blocks.append(w_deq)

        if self.use_blockwise:
            padded_out = torch.zeros_like(self.padded_weight)
            for idx, (i, j) in enumerate(self.block_metadata):
                padded_out[i:i+self.block_size, j:j+self.block_size] = q_blocks[idx]
            return padded_out[:self.original_shape[0], :self.original_shape[1]].cpu()
        else:
            return q_blocks[0].cpu()


class ColumnwiseQuantizationOptim(nn.Module):
    def __init__(self, weight: torch.Tensor, num_bits: int = 4, fixed_T: float = 100.0):
        super().__init__()
        self.num_bits = num_bits
        self.fixed_T = fixed_T
        self.num_levels = 2 ** num_bits
        self.original_shape = weight.shape  # [rows, cols]

        self.w_min = nn.ParameterList()
        self.w_max = nn.ParameterList()

        for j in range(weight.size(1)):  # Iterate over columns
            col = weight[:, j]
            w_min = col.min().detach()
            w_max = col.max().detach()
            pad = 0.01 * (w_max - w_min)
            self.w_min.append(nn.Parameter((w_min - pad).view(1)))
            self.w_max.append(nn.Parameter((w_max + pad).view(1)))

    def forward(self, weight: torch.Tensor):
        eps = 1e-6
        rows, cols = weight.shape

        w_dequant = torch.zeros_like(weight)
        total_entropy = 0.0

        q_levels = torch.linspace(0, 1, self.num_levels, device=weight.device)

        for j in range(cols):
            col = weight[:, j]
            w_min = self.w_min[j].clamp(max=self.w_max[j].item() - eps)
            w_max = self.w_max[j].clamp(min=w_min.item() + eps)

            w_norm = (col - w_min) / (w_max - w_min + eps)
            dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
            soft_probs = torch.softmax(dists * self.fixed_T, dim=-1)
            w_q = (soft_probs * q_levels).sum(dim=-1)
            w_deq = w_q * (w_max - w_min) + w_min
            w_dequant[:, j] = w_deq

            bin_mass = soft_probs.sum(dim=0)
            bin_probs = bin_mass / (bin_mass.sum() + eps)
            entropy = -(bin_probs * (bin_probs + eps).log()).sum()
            total_entropy += entropy

        return w_dequant, total_entropy

    def export(self, weight: torch.Tensor):
        eps = 1e-6
        rows, cols = weight.shape

        q_indices = []
        dequant_out = torch.zeros_like(weight)
        q_levels = torch.linspace(0, 1, self.num_levels, device=weight.device)

        for j in range(cols):
            col = weight[:, j]
            w_min = self.w_min[j].clamp(max=self.w_max[j].item() - eps)
            w_max = self.w_max[j].clamp(min=w_min.item() + eps)

            w_norm = (col - w_min) / (w_max - w_min + eps)
            dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
            q_idx = torch.argmax(dists, dim=-1).to(torch.int32)
            w_q = q_levels[q_idx]
            w_deq = w_q * (w_max - w_min) + w_min

            q_indices.append(q_idx.cpu())
            dequant_out[:, j] = w_deq

        return {
            "q_indices_columns": q_indices,
            "w_min": torch.stack([p.detach().cpu() for p in self.w_min]),
            "w_max": torch.stack([p.detach().cpu() for p in self.w_max]),
            "dequant": dequant_out.cpu()
        }


# === Quantization Loop for all Linear Layers ===
safetensor_dict = {}
flag = 0
for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    if "lm_head" in name:
        continue
    # if "fc1" in name:
    #     continue

    if "embed_out" in name:
        continue


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
    original_weight = module.weight.data.clone()
    # gptq = GPTQ(module)
    # for act in activation_batches:
    #     gptq.add_batch(act, module(act))
    # gptq.fasterquant(
    #     blocksize=BLOCK_SIZE,
    #     percdamp=0.01,
    #     group_size=128,
    #     actorder=True,
    # )
    # q_weight = module.weight.data.clone()

    #quant_layer = BlockwiseQuantizationOptim(original_weight,num_bits = 4, use_blockwise=False,a=0.55).to(DEVICE)
    quant_layer = ColumnwiseQuantizationOptim(original_weight, num_bits=4).to(DEVICE)

    optimizer = torch.optim.Adam(quant_layer.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    #original_weight = module.weight.data.clone()

    for it in range(NUM_ITERATIONS):
        for act in activation_batches:
            optimizer.zero_grad()
            w_q, entropy = quant_layer(original_weight)
            recon = F.linear(act.to(DEVICE), w_q)
            target = F.linear(act.to(DEVICE), original_weight)
            loss = mse_loss(recon, target) + mse_loss(original_weight, w_q)
            print(f"Iteration {it + 1}/{NUM_ITERATIONS}, Entropy: {entropy.item():.4f}, Loss: {loss.item():.8f}")
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_weight = quant_layer.export(original_weight)["dequant"].to(module.weight.device)
        loss = mse_loss(original_weight, final_weight)
        print("weight diff",loss)
        module.weight.copy_(final_weight)
        #safetensor_dict[name.replace(".", "_") + ".dequant"] = final_weight
    del quant_layer, optimizer, activation_batches
    torch.cuda.empty_cache()

# === Save Final Weights ===
#save_file(safetensor_dict, "quantized_blockwise_gptq.safetensors")
print("\n✅ Finished GPTQ-initialized blockwise quantization for all layers.")
