# === Setup ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
#from gptq import GPTQ
import math

# === CONFIG ===
MODEL_NAME = "facebook/opt-350m"
#MODEL_NAME = "databricks/dolly-v2-3b"
#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat"
MODEL_NAME = "openlm-research/open_llama_3b_v2"
#MODEL_NAME = "NousResearch/Meta-Llama-3-8B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
N_BATCHES = 1
SEQ_LEN = 32
NUM_BITS = 4
BLOCK_SIZE = 128
FIXED_T = 1000.0
LR = 0.001
NUM_ITERATIONS = 0

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
            pad = 0.00 * (w_max - w_min)
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

        return w_dequant, total_entropy/cols

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
        # === Pack two 4-bit indices into a single uint8 value ===
        q_indices = torch.stack(q_indices, dim=1)  # Shape: [rows, cols]
        flat = q_indices.flatten().to(torch.uint8)  # 🔁 Force dtype
        if flat.numel() % 2 != 0:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
        packed = (flat[0::2] << 4) | (flat[1::2])
        packed = packed.to(torch.uint8).cpu()  # 🔁 Make sure it's uint8

        #return dequant_out
        return {
            "packed_q_indices_4bit": packed,
            "q_indices_columns": q_indices,
            "w_min": torch.stack([p.detach().cpu() for p in self.w_min]),
            "w_max": torch.stack([p.detach().cpu() for p in self.w_max]),
            "dequant": dequant_out#.cpu()
        }
skip_layers = ["model.layers.1.mlp.down_proj","model.layers.2.mlp.down_proj"]

# === Quantization Loop for all Linear Layers ===
safetensor_dict = {}
flag = 0
for name, module in model.named_modules():
    # === Handle Non-Quantized Layers (Norms, Embeddings, Bias) ===
    if not isinstance(module, nn.Linear):
        print("non linear layer: ",name)
        if isinstance(module, (nn.LayerNorm, nn.Embedding)):
            if hasattr(module, "weight"):
                safetensor_dict[f"{name.replace('.', '_')}_weight"] = module.weight.data.clone().cpu()
            if hasattr(module, "bias") and module.bias is not None:
                safetensor_dict[f"{name.replace('.', '_')}_bias"] = module.bias.data.clone().cpu()
        continue
    
    if "lm_head" in name:
        if isinstance(module, nn.Linear):
            # If not quantized, export full fp32
            safetensor_dict[f"{name.replace('.', '_')}_weight_fp32"] = module.weight.data.clone().cpu()
            if module.bias is not None:
                safetensor_dict[f"{name.replace('.', '_')}_bias_fp32"] = module.bias.data.clone().cpu()
        continue
    
    if name in skip_layers:
        if isinstance(module, nn.Linear):
            # If not quantized, export full fp32
            safetensor_dict[f"{name.replace('.', '_')}_weight_fp32"] = module.weight.data.clone().cpu()
            if module.bias is not None:
                safetensor_dict[f"{name.replace('.', '_')}_bias_fp32"] = module.bias.data.clone().cpu()
        continue

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
    # if flag == 10:
    #   break
    # flag += 1
    #quant_layer = BlockwiseQuantizationOptim(original_weight,num_bits = 4, use_blockwise=True,a=1).to(DEVICE)
    quant_layer = ColumnwiseQuantizationOptim(original_weight, num_bits=4).to(DEVICE)

    # === Stack all activations once ===
    activation_all = torch.cat(activation_batches, dim=0)  # Shape: [BATCH_SIZE * N_BATCHES, SEQ_LEN, hidden]
    activation_all = activation_all.to(DEVICE)

    # === Precompute target using original weights ===
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        target = F.linear(activation_all, original_weight)

    # === Define fused training step ===
    def quant_train_step(quant_layer, weight, act, target):
        w_q, entropy = quant_layer(weight)
        recon = F.linear(act, w_q)
        loss = mse_loss(recon, target) + mse_loss(weight, w_q) + 0.001 * entropy
        return loss, entropy

    # Optional Torch Compile (PyTorch 2+ only)
    quant_train_step = torch.compile(quant_train_step, mode="reduce-overhead")

    optimizer = torch.optim.Adam(quant_layer.parameters(), lr=LR)
    mse_loss = nn.MSELoss()

    # === Training loop ===
    for it in range(NUM_ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, entropy = quant_train_step(quant_layer, original_weight, activation_all, target)
        loss.backward()
        optimizer.step()
        print(f"🌀 Iter {it + 1}/{NUM_ITERATIONS} | Entropy: {entropy.item():.4f} | Loss: {loss.item():.6f}")


    with torch.no_grad():
        layer_key = name.replace(".", "_")
        export_data = quant_layer.export(original_weight)

        final_weight = export_data["dequant"].to(module.weight.device)
        loss = mse_loss(original_weight, final_weight)
        print("weight diff", loss)
        module.weight.copy_(final_weight)

        # Save quantization outputs to safetensors
        safetensor_dict[f"{layer_key}_dequant"] = final_weight
        safetensor_dict[f"{layer_key}_packed_q4_indices"] = export_data["packed_q_indices_4bit"]
        safetensor_dict[f"{layer_key}_w_min"] = export_data["w_min"]
        safetensor_dict[f"{layer_key}_w_max"] = export_data["w_max"]

        # Optional: save raw q_indices if needed later
        # safetensor_dict[f"{layer_key}_q_indices"] = export_data["q_indices_columns"]

        # Also save bias if present
        if module.bias is not None:
            safetensor_dict[f"{layer_key}_bias_fp32"] = module.bias.data.clone().cpu()
           
    prompt = "I like travelling to"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=50)

    print("Sample Output:", tokenizer.decode(output[0], skip_special_tokens=True))
    del quant_layer, optimizer, activation_batches
    torch.cuda.empty_cache()

# === Save Final Weights ===
save_file(safetensor_dict, "quantized_blockwise_gptq.safetensors")
print("\n✅ Finished GPTQ-initialized blockwise quantization for all layers.")
import zstandard as zstd
import torch
from safetensors.torch import load_file

def compress_tensor_to_zstd_size(tensor: torch.Tensor) -> int:
    print(tensor.dtype)
    assert tensor.dtype == torch.uint8, "Expected packed 4-bit indices as uint8."
    data_bytes = tensor.cpu().numpy().tobytes()
    cctx = zstd.ZstdCompressor(level=10)
    compressed = cctx.compress(data_bytes)
    return len(compressed)

def analyze_quantized_compression(safetensor_path: str):
    data = load_file(safetensor_path)
    
    total_original_size = 0
    total_compressed_size = 0

    print(f"🔍 Analyzing compression in: {safetensor_path}\n")

    for key in data:
        if key.endswith("_packed_q4_indices"):
            base_name = key.replace("_packed_q4_indices", "")
            q4_tensor = data[key]

            shape = data.get(f"{base_name}_dequant").size()
            #print(shape_tensor)
            # w_max = data.get(f"{base_name}_w_max")
            # shape_tensor = data.get(f"{base_name}_shape")

            #assert shape_tensor is not None, f"Missing shape for {base_name}"
            #shape = tuple(shape_tensor.tolist())
            num_elements = shape[0] * shape[1]

            # === Original FP32 weight size ===
            original_size_bytes = num_elements * 4  # float32 = 4 bytes

            # === Compressed size via Zstd ===
            compressed_size_bytes = compress_tensor_to_zstd_size(q4_tensor)

            print(f"📦 Layer: {base_name}")
            print(f"   Original size:   {original_size_bytes / 1024:.2f} KB")
            print(f"   Compressed size: {compressed_size_bytes / 1024:.2f} KB")
            print(f"   Compression ratio: {original_size_bytes / compressed_size_bytes:.2f}x\n")

            total_original_size += original_size_bytes
            total_compressed_size += compressed_size_bytes

    print("=== 📊 Summary ===")
    print(f"Total original size:   {total_original_size / (1024 * 1024):.2f} MB")
    print(f"Total compressed size: {total_compressed_size / (1024 * 1024):.2f} MB")
    print(f"Overall compression ratio: {total_original_size / total_compressed_size:.2f}x")

# === Example usage ===
if __name__ == "__main__":
    analyze_quantized_compression("quantized_blockwise_gptq.safetensors")
