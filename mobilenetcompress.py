import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from safetensors.torch import save_file
import resnet
import mobilenetv2

# ==== Config ====
BATCH_SIZE = 128
NUM_BITS = 8
FIXED_T = 100.5
LR = 0.001
NUM_ITERATIONS = 100
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
#model = resnet.resnet18(pretrained=False, device=DEVICE).to(DEVICE)
model = mobilenetv2.mobilenet_v2(pretrained=False, device=DEVICE).to(DEVICE)
state_dict = torch.load("/content/mobilenet_v2.pt", map_location="cpu")
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

# ==== Optimization & Export ====
def optimize_all_layers(model, test_loader):
    model.eval()
    data_iterator = iter(test_loader)
    x, y = next(data_iterator)
    x, y = x[:128].to(DEVICE), y[:128].to(DEVICE)

    with torch.no_grad():
        model(x)

    safetensor_dict = {}

    for name, param in model.named_parameters():
        module_name = name.rsplit('.', 1)[0]
        try:
            mod = dict(model.named_modules())[module_name]
            if isinstance(mod, nn.BatchNorm2d) or name.endswith(".bias"):
                base_name = name.replace('.', '_')
                safetensor_dict[f"{base_name}.dequant"] = param.detach().cpu()
                print(f"🟢 Stored unquantized: {name}")
        except KeyError:
            print(f"⚠️ Module not found for param: {name}")

        print(f"\n🔧 Optimizing {name}...")
        layer_name = name.replace(".weight", "")
        matched_key = next((k for k in temp_activations if layer_name in k), None)
        activation_input = temp_activations.get(matched_key, None)
        if activation_input is not None:
            activation_input = activation_input.detach().clone().to(DEVICE)
            del temp_activations[matched_key]

        original_param = param.detach().clone()
        entropy_budget = (32 * original_param.numel()) / CR_target
        quant_layer = MinMaxQuantization(original_param, entropy_budget=entropy_budget).to(DEVICE)
        optimizer = optim.Adam(quant_layer.parameters(), lr=LR)
        mse_loss_fn = nn.MSELoss()
        original_param_data = param.data.clone()

        for iteration in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            q_param, entropy, penalty, soft_probs = quant_layer(original_param)
            recon_loss = torch.tensor(0.0, device=DEVICE)

            if activation_input is not None:
                try:
                    conv_layer = next((m for n, m in model.named_modules()
                                       if n == layer_name and isinstance(m, (nn.Conv2d, nn.Linear))), None)
                    if isinstance(conv_layer, nn.Conv2d):
                        q_out = nn.functional.conv2d(activation_input, q_param,
                                                     stride=conv_layer.stride, padding=conv_layer.padding,
                                                     dilation=conv_layer.dilation, groups=conv_layer.groups)
                        o_out = nn.functional.conv2d(activation_input, original_param,
                                                     stride=conv_layer.stride, padding=conv_layer.padding,
                                                     dilation=conv_layer.dilation, groups=conv_layer.groups)
                        recon_loss = mse_loss_fn(q_out, o_out)
                    elif isinstance(conv_layer, nn.Linear):
                        flat = activation_input.view(activation_input.size(0), -1)
                        q_out = nn.functional.linear(flat, q_param)
                        o_out = nn.functional.linear(flat, original_param)
                        recon_loss = mse_loss_fn(q_out, o_out)
                except:
                    recon_loss = torch.tensor(0.0, device=DEVICE)

            with torch.no_grad():
                param.data = q_param.detach()
                class_loss = nn.CrossEntropyLoss()(model(x), y)
                param.data = original_param_data

            # if class_loss > 0.2:
            #     break

            total_loss = 0.1 * recon_loss + 0.9 * class_loss + 0.0 * entropy
            total_loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"Iter {iteration}: recon={recon_loss.item():.6f}, class={class_loss.item():.4f}, entropy={entropy.item():.2f}")

        final_param = quant_layer.export_hard_quant(original_param)
        with torch.no_grad():
            param.copy_(final_param["dequant"].to(param.device))

        base_name = name.replace('.', '_')
        safetensor_dict[f"{base_name}.q_indices"] = final_param["q_indices"]
        safetensor_dict[f"{base_name}.w_min"] = final_param["w_min"]
        safetensor_dict[f"{base_name}.w_max"] = final_param["w_max"]
        safetensor_dict[f"{base_name}.dequant"] = final_param["dequant"]

        torch.cuda.empty_cache()
        with torch.no_grad():
            model(x)

    # Store skipped biases and BN params as-is
    for name, param in model.named_parameters():
        if "bn" in name or ".bias" in name:
            base_name = name.replace('.', '_')
            safetensor_dict[f"{base_name}.dequant"] = param.detach().cpu()
            print(f"🟢 Stored unquantized: {name}")

    save_file(safetensor_dict, "quantized_model.safetensors")
    print("💾 Saved all quantized weights to 'quantized_model.safetensors'")
    print("✅ Full-layer quantization complete.")

# ==== Run ====
print("\n📊 Accuracy BEFORE quantization:")
evaluate(model, test_loader)
optimize_all_layers(model, test_loader)
print("\n📊 Accuracy AFTER quantization:")
evaluate(model, test_loader)
