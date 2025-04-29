import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from safetensors.torch import save_file
import resnet
#import mobilenetv2

# ==== Config ====
BATCH_SIZE = 128
NUM_BITS = 8
FIXED_T = 100.5
LR = 0.001
NUM_ITERATIONS = 30
CR_target = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Load model ====
model = resnet.resnet18(pretrained=False, device=DEVICE).to(DEVICE)
#model = mobilenetv2.mobilenet_v2(pretrained=False, device=DEVICE).to(DEVICE)
state_dict = torch.load("/content/resnet18.pt", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# ==== Evaluation ====
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            _, p = torch.max(pred, 1)
            correct += (p == y).sum().item()
            total += y.size(0)
    acc = 100 * correct / total
    print(f"🌟 Accuracy: {acc:.2f}%")
    return acc

# ==== Activation Capture ====
temp_activations = {}
def activation_hook(layer_name):
    def hook(module, input, output):
        temp_activations[layer_name] = input[0].detach().clone()
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.register_forward_hook(activation_hook(name))

# ==== Quantizer ====
class MinMaxQuantization(nn.Module):
    def __init__(self, weight, num_levels=2**NUM_BITS, fixed_T=FIXED_T, entropy_budget=None):
        super().__init__()
        self.num_levels = num_levels
        self.fixed_T = fixed_T
        self.entropy_budget = entropy_budget
        w_min_init = weight.min().detach()
        w_max_init = weight.max().detach()
        pad = 0.05 * (w_max_init - w_min_init)
        self.w_min = nn.Parameter(w_min_init - pad)
        self.w_max = nn.Parameter(w_max_init + pad)

    def forward(self, w):
        EPS = 1e-6
        w_min = self.w_min.clamp(max=self.w_max.item() - EPS)
        w_max = self.w_max.clamp(min=w_min.item() + EPS)
        w_norm = (w - w_min) / (w_max - w_min + EPS)
        q_levels = torch.linspace(0, 1, self.num_levels, device=w.device)
        dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
        soft_probs = torch.softmax(dists * self.fixed_T, dim=-1)
        w_q = (soft_probs * q_levels).sum(dim=-1)
        w_deq = w_q * (w_max - w_min) + w_min
        bin_mass = soft_probs.sum(dim=0)
        bin_probs = bin_mass / bin_mass.sum()
        entropy = -(bin_probs * (bin_probs + EPS).log()).sum()
        budget_penalty = (entropy / (self.entropy_budget + EPS)) ** 2
        return w_deq, entropy, budget_penalty, soft_probs

    def export_hard_quant(self, w):
        EPS = 1e-6
        w_min = self.w_min.clamp(max=self.w_max.item() - EPS)
        w_max = self.w_max.clamp(min=w_min.item() + EPS)
        w_norm = (w - w_min) / (w_max - w_min + EPS)
        q_levels = torch.linspace(0, 1, self.num_levels, device=w.device)
        dists = -torch.abs(w_norm.unsqueeze(-1) - q_levels)
        q_indices = torch.argmax(dists, dim=-1).clamp(0, self.num_levels - 1).to(torch.int32)
        w_q = q_levels[q_indices]
        w_deq = w_q * (w_max - w_min) + w_min
        return {
            "q_indices": q_indices.cpu(),
            "w_min": w_min.cpu().unsqueeze(0),
            "w_max": w_max.cpu().unsqueeze(0),
            "dequant": w_deq.cpu()
        }

def optimize_all_layers(model, test_loader):
    model.eval()
    N_BATCHES = 10  # Number of batches to process
    BATCH_SIZE = 128

    safetensor_dict = {}
    named_modules = dict(model.named_modules())

    # Start optimizing each layer
    for name, param in model.named_parameters():
        module_name = name.rsplit('.', 1)[0]
        try:
            mod = named_modules[module_name]
            if isinstance(mod, nn.BatchNorm2d) or name.endswith(".bias"):
                base_name = name.replace('.', '_')
                safetensor_dict[f"{base_name}.dequant"] = param.detach().cpu()
                print(f"🟢 Stored unquantized: {name}")
                continue
        except KeyError:
            print(f"⚠️ Module not found for param: {name}")

        print(f"\n🔧 Optimizing {name}...")

        # === Step 1: Capture this layer's activations batch-by-batch ===
        activation_batches = []  # Store activations batch-by-batch
        captured_activations = []

        def capture_hook(module, input, output):
            captured_activations.append(input[0].detach().cpu())

        hook = None
        try:
            module = named_modules.get(module_name, None)
            if module is not None:
                hook = module.register_forward_hook(capture_hook)

            # Run N_BATCHES separately
            data_iterator = iter(test_loader)
            for _ in range(N_BATCHES):
                try:
                    x, y = next(data_iterator)
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    captured_activations = []
                    with torch.no_grad():
                        model(x)  # Will populate captured_activations
                    if len(captured_activations) > 0:
                        activation_batches.append(captured_activations[0])  # Only input to this layer
                    else:
                        print(f"⚠️ No activation captured for batch.")
                except StopIteration:
                    break

            if hook:
                hook.remove()

            if len(activation_batches) == 0:
                print(f"⚠️ No activations captured for {name}, skipping.")
                continue

            print(f"✅ Captured {len(activation_batches)} activation batches for {name}.")

        except Exception as e:
            print(f"❌ Error capturing activations for {name}: {e}")
            continue

        # === Step 2: Start optimization using stored small activations ===
        original_param = param.detach().clone()
        entropy_budget = (32 * original_param.numel()) / CR_target
        quant_layer = MinMaxQuantization(original_param, entropy_budget=entropy_budget).to(DEVICE)
        optimizer = optim.Adam(quant_layer.parameters(), lr=LR)
        mse_loss_fn = nn.MSELoss()
        original_param_data = param.data.clone()

        for iteration in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            recon_loss_total = 0.0
            class_loss_total = 0.0
            q_param, entropy, penalty, soft_probs = quant_layer(original_param)

            # Replay batches one-by-one
            data_iterator = iter(test_loader)
            for idx, activation_input_cpu in enumerate(activation_batches):
                try:
                    x_batch, y_batch = next(data_iterator)
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                except StopIteration:
                    break

                activation_input = activation_input_cpu.to(DEVICE)

                recon_loss = torch.tensor(0.0, device=DEVICE)
                if isinstance(module, nn.Conv2d):
                    q_out = nn.functional.conv2d(activation_input, q_param,
                                                 stride=module.stride, padding=module.padding,
                                                 dilation=module.dilation, groups=module.groups)
                    o_out = nn.functional.conv2d(activation_input, original_param,
                                                 stride=module.stride, padding=module.padding,
                                                 dilation=module.dilation, groups=module.groups)
                    recon_loss = mse_loss_fn(q_out, o_out)

                elif isinstance(module, nn.Linear):
                    flat = activation_input.view(activation_input.size(0), -1)
                    q_out = nn.functional.linear(flat, q_param)
                    o_out = nn.functional.linear(flat, original_param)
                    recon_loss = mse_loss_fn(q_out, o_out)

                with torch.no_grad():
                    param.data = q_param.detach()
                    outputs = model(x_batch)
                    class_loss = nn.CrossEntropyLoss()(outputs, y_batch)
                    param.data = original_param_data

                recon_loss_total += recon_loss
                class_loss_total += class_loss

            recon_loss_avg = recon_loss_total / len(activation_batches)
            class_loss_avg = class_loss_total / len(activation_batches)

            total_loss = 0.1 * recon_loss_avg + 0.9 * class_loss_avg + 0.0 * entropy
            total_loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"Iter {iteration}: recon={recon_loss_avg.item():.6f}, "
                      f"class={class_loss_avg.item():.4f}, entropy={entropy.item():.2f}")

        # === Step 3: Save optimized quantized weights ===
        final_param = quant_layer.export_hard_quant(original_param)
        with torch.no_grad():
            param.copy_(final_param["dequant"].to(param.device))

        base_name = name.replace('.', '_')
        safetensor_dict[f"{base_name}.q_indices"] = final_param["q_indices"]
        safetensor_dict[f"{base_name}.w_min"] = final_param["w_min"]
        safetensor_dict[f"{base_name}.w_max"] = final_param["w_max"]
        safetensor_dict[f"{base_name}.dequant"] = final_param["dequant"]

        # Cleanup memory
        del activation_batches
        torch.cuda.empty_cache()

    # Save BatchNorm running_mean and running_var
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            base_name = name.replace('.', '_')
            safetensor_dict[f"{base_name}.running_mean"] = module.running_mean.detach().cpu()
            safetensor_dict[f"{base_name}.running_var"] = module.running_var.detach().cpu()
            print(f"🟢 Stored BN running stats: {name}")

    save_file(safetensor_dict, "quantized_model.safetensors")
    print("💾 Saved all quantized weights to 'quantized_model.safetensors'")
    print("✅ Full-layer quantization complete.")



# ==== Run ====
print("\n📊 Accuracy BEFORE quantization:")
#evaluate(model, test_loader)
optimize_all_layers(model, test_loader)
print("\n📊 Accuracy AFTER quantization:")
evaluate(model, test_loader)
