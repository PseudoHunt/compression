import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import math

# ==== CONFIG ====
MODEL_NAME = "facebook/opt-125m"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
N_BATCHES = 5
SEQ_LEN = 32
NUM_BITS = 4
FIXED_T = 100.0
LR = 0.001
NUM_ITERATIONS = 100
CR_target = 100

# ==== Load model and tokenizer ====
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==== Multi-Prompt Setup ====
base_prompts = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, a battle was brewing.",
    "Once upon a time in a small village,",
    "Artificial intelligence is transforming the world.",
    "Python is a great language for machine learning."
]

SKIP_LAYERS = {
    "model.decoder.layers.11.fc2",
    "model.decoder.layers.10.fc2",
}




all_prompts = base_prompts * ((BATCH_SIZE * N_BATCHES) // len(base_prompts) + 1)
all_prompts = all_prompts[:BATCH_SIZE * N_BATCHES]

inputs = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN)
input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs["attention_mask"].to(DEVICE)
input_batches = input_ids.split(BATCH_SIZE)
mask_batches = attention_mask.split(BATCH_SIZE)

# --- Blockwise Quantization with Optimization ---
class BlockwiseQuantizationOptim(nn.Module):
    def __init__(self, weight: torch.Tensor, block_size: int = 128, num_bits: int = 8, fixed_T: float = 100.0):
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
                self.blocks.append(self.padded_weight[i:i+block_size, j:j+block_size])
                self.block_metadata.append((i, j))

        self.w_min = nn.ParameterList()
        self.w_max = nn.ParameterList()
        for block in self.blocks:
            w_min, w_max = block.min().detach(), block.max().detach()
            pad = 0.00 * (w_max - w_min)
            self.w_min.append(nn.Parameter((w_min - pad).view(1)))
            self.w_max.append(nn.Parameter((w_max + pad).view(1)))

    def forward(self):
        q_blocks = []
        total_entropy = 0.0
        eps = 1e-6
        for idx, block in enumerate(self.blocks):
            w_min = self.w_min[idx].clamp(max=self.w_max[idx].item() - eps)
            w_max = self.w_max[idx].clamp(min=w_min.item() + eps)

            w_norm = (block - w_min) / (w_max - w_min + eps)
            q_levels = torch.linspace(0, 1, self.num_levels, device=block.device)
            dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
            soft_probs = torch.softmax(dists * self.fixed_T, dim=-1)
            w_q = (soft_probs * q_levels).sum(dim=-1)
            w_deq = w_q * (w_max - w_min) + w_min

            bin_mass = soft_probs.sum(dim=0)
            bin_probs = bin_mass / (bin_mass.sum() + eps)
            entropy = -(bin_probs * (bin_probs + eps).log()).sum()
            total_entropy += entropy

            q_blocks.append(w_deq)

        padded_out = torch.zeros_like(self.padded_weight)
        for idx, (i, j) in enumerate(self.block_metadata):
            padded_out[i:i+self.block_size, j:j+self.block_size] = q_blocks[idx]

        return padded_out[:self.original_shape[0], :self.original_shape[1]], total_entropy

    def export(self):
        q_blocks = []
        q_indices_list, w_min_list, w_max_list = [], [], []
        eps = 1e-6
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
            q_indices_list.append(q_idx.cpu())
            w_min_list.append(w_min.view(1).cpu())
            w_max_list.append(w_max.view(1).cpu())

        padded_out = torch.zeros_like(self.padded_weight)
        for idx, (i, j) in enumerate(self.block_metadata):
            padded_out[i:i+self.block_size, j:j+self.block_size] = q_blocks[idx]

        full_weight = padded_out[:self.original_shape[0], :self.original_shape[1]]
        return {
            "q_indices_blocks": q_indices_list,
            "w_min_blocks": w_min_list,
            "w_max_blocks": w_max_list,
            "dequant": full_weight.cpu()
        }

# ==== Quantization Loop ====
safetensor_dict = {}
named_modules = dict(model.named_modules())

for module_name, module in model.named_modules():
    # Skip non-linear layers
    if not isinstance(module, nn.Linear):
        continue

    if module_name in SKIP_LAYERS:
        print(f"⏭️ Skipping layer: {module_name}")
        continue

    # Skip lm_head
    if module is getattr(model, "lm_head", None):
        print(f"⏭️ Skipping lm_head: {module_name}")
        continue

    weight = module.weight.detach()
    print(f"\n🔧 Optimizing: {module_name} | Shape: {weight.shape}")

    activation_batches = []

    def capture_hook(mod, input, output):
        activation_batches.append(input[0].detach().cpu())

    hook = module.register_forward_hook(capture_hook)
    with torch.no_grad():
        for x_batch, m_batch in zip(input_batches, mask_batches):
            model(input_ids=x_batch, attention_mask=m_batch)
    hook.remove()

    if not activation_batches:
        print(f"⚠️ No activations for {module_name}, skipping.")
        continue

    activation_batches = [a.to(DEVICE) for a in activation_batches]
    quant_layer = BlockwiseQuantizationOptim(weight, block_size=128, num_bits=NUM_BITS).to(weight.device)
    optimizer = torch.optim.Adam(quant_layer.parameters(), lr=LR)
    mse_loss = nn.MSELoss()
    original_weight = module.weight.data.clone()

    # Step 2: Compute initial class loss once for the layer
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        labels = input_ids[:, 1:]
        logits = outputs.logits[:, :-1, :]
        base_class_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

    # === Optimization loop ===
    for iteration in range(NUM_ITERATIONS):
        for activation in activation_batches:
            optimizer.zero_grad()
            q_weight, entropy = quant_layer()

            # Recon loss
            recon_loss = mse_loss(
                nn.functional.linear(activation, q_weight),
                nn.functional.linear(activation, original_weight)
            )

            # Patch weight temporarily for class loss
            with torch.no_grad():
                module.weight.copy_(q_weight)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                labels = input_ids[:, 1:]
                logits = outputs.logits[:, :-1, :]
                class_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

                module.weight.copy_(original_weight)

            total_loss = 0.9 * recon_loss + 0.1 * class_loss
            total_loss.backward()
            optimizer.step()

            del q_weight
            torch.cuda.empty_cache()

        if iteration % 10 == 0:
            print(f"Iter {iteration}: recon={recon_loss.item():.6f}, class={class_loss.item():.4f}, entropy={entropy.item():.2f}")


    export = quant_layer.export()
    base = module_name.replace(".", "_")
    # safetensor_dict[f"{base}_q_indices"] = export["q_indices"]
    # safetensor_dict[f"{base}_w_min"] = export["w_min"]
    # safetensor_dict[f"{base}_w_max"] = export["w_max"]
    # safetensor_dict[f"{base}_dequant"] = export["dequant"]

    with torch.no_grad():
        module.weight.copy_(export["dequant"].to(module.weight.device))

    # Cleanup
    del activation_batches, quant_layer, optimizer, export, original_weight
    torch.cuda.empty_cache()

# ==== Save weights ====
save_file(safetensor_dict, "quantized_llm_model.safetensors")
print("\n✅ Quantization complete. Saved to quantized_llm_model.safetensors")
