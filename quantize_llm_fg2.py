# === FG2 Hessian-Enhanced Quantization Script (OPT-125M) ===

import torch
import torch.nn as nn
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

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

# ==== Load model ====
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==== Prompts ====
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


# === FG2 Hessian Computation ===
def compute_fg2_hessian_diag(module, model, input_ids, attention_mask, tokenizer,
                              activation_batches, original_weight, loss_fn, eps=1e-2):
    hessian_diag = torch.zeros_like(original_weight)
    original_weight = original_weight.clone().detach()

    for act in activation_batches:
        perturb = torch.randn_like(original_weight) * eps

        def compute_loss(perturbed_weight):
            with torch.no_grad():
                module.weight.copy_(original_weight + perturbed_weight)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:]
                return loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        loss_plus = compute_loss(+perturb)
        loss_minus = compute_loss(-perturb)
        loss_center = compute_loss(torch.zeros_like(original_weight))

        hess_estimate = (loss_plus + loss_minus - 2 * loss_center) / (eps ** 2)
        hessian_diag += hess_estimate * perturb.sign()

    hessian_diag = hessian_diag.abs()
    hessian_diag /= hessian_diag.max() + 1e-6
    return hessian_diag.cpu()


# === Quantization Layer ===
class BlockwiseQuantizationOptim(nn.Module):
    def __init__(self, weight, block_size=128, num_bits=4, fixed_T=100.0):
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
            pad = 0.01 * (w_max - w_min)
            self.w_min.append(nn.Parameter((w_min + pad).view(1)))
            self.w_max.append(nn.Parameter((w_max - pad).view(1)))

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

# === FG2 Hessian-Enhanced Quantization Script (OPT-125M) ===



# === Quantization Loop ===
safetensor_dict = {}
for module_name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    if module_name in SKIP_LAYERS:
        print(f"⏭️ Skipping layer: {module_name}")
        continue
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
    original_weight = module.weight.data.clone()

    # Compute FG2 Hessian
    fg2_hessian = compute_fg2_hessian_diag(
        module=module,
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        activation_batches=activation_batches,
        original_weight=original_weight,
        loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    )

    quant_layer = BlockwiseQuantizationOptim(weight, block_size=128, num_bits=NUM_BITS).to(weight.device)
    optimizer = torch.optim.Adam(quant_layer.parameters(), lr=LR)
    mse_loss = nn.MSELoss(reduction='none')  # use elementwise loss

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        labels = input_ids[:, 1:]
        logits = outputs.logits[:, :-1, :]
        base_class_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

    for iteration in range(NUM_ITERATIONS):
        for activation in activation_batches:
            optimizer.zero_grad()
            q_weight, entropy = quant_layer()

            # Weighted reconstruction loss using FG2 Hessian importance
            with torch.no_grad():
                weight_importance = fg2_hessian.to(q_weight.device).view(1,-1)
                

            recon_elementwise = mse_loss(
                nn.functional.linear(activation, q_weight),
                nn.functional.linear(activation, original_weight)
            )
            weighted_recon_loss = (recon_elementwise * weight_importance.mean(dim=1, keepdim=True)).mean()

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

            total_loss = 0.9 * weighted_recon_loss + 0.1 * class_loss
            total_loss.backward()
            optimizer.step()
            del q_weight
            torch.cuda.empty_cache()

        if iteration % 10 == 0:
            print(f"Iter {iteration}: recon={weighted_recon_loss.item():.6f}, class={class_loss.item():.4f}, entropy={entropy.item():.2f}")

    export = quant_layer.export()
    base = module_name.replace(".", "_")
    with torch.no_grad():
        module.weight.copy_(export["dequant"].to(module.weight.device))

    del activation_batches, quant_layer, optimizer, export, original_weight
    torch.cuda.empty_cache()

# ==== Save weights ====
save_file(safetensor_dict, "quantized_llm_model.safetensors")
print("\n✅ Quantization complete. Saved to quantized_llm_model.safetensors")
