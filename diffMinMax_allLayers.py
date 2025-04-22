import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np
import resnet  # Your local resnet18 module

# ==== Hyperparameters ====
BATCH_SIZE = 128
NUM_BITS = 4
FIXED_T = 100.5
LR = 0.001
NUM_ITERATIONS = 100
CR_target = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Model ====
model = resnet.resnet18(pretrained=False, device=device).to(device)
state_dict = torch.load('/content/resnet18.pt', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# ==== Evaluation ====
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"🎯 Accuracy: {100 * correct / total:.2f}%")

# ==== Activation Hook ====
temp_activations = {}
def activation_hook(layer_name):
    def hook(module, input, output):
        temp_activations[layer_name] = input[0].detach().clone()
    return hook

# Register hooks for Conv2d and Linear layers only
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.register_forward_hook(activation_hook(name))

layer_soft_probs = {}

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

# ==== Optimization ====
def optimize_all_layers(model, test_loader):
    model.eval()
    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images, labels = images[:2].to(device), labels[:2].to(device)

    with torch.no_grad():
        model(images)

    for name, param in model.named_parameters():
        if not any(x in name for x in ['weight', 'bias']):
            continue

        print(f"\n🔧 Optimizing {name}...")
        layer_name = name.replace(".weight", "").replace(".bias", "")

        matched_key = next((k for k in temp_activations if layer_name in k), None)
        if matched_key:
            activation_input = temp_activations[matched_key].detach().clone().to(device)
            del temp_activations[matched_key]
        else:
            print(f"⚠️ No activation found for {name} — recon_loss will be skipped.")
            activation_input = None

        original_param = param.detach().clone()
        num_weights = original_param.numel()
        entropy_budget = (32 * num_weights) / CR_target

        quant_layer = MinMaxQuantization(original_param, entropy_budget=entropy_budget).to(device)
        optimizer = optim.Adam(quant_layer.parameters(), lr=LR)
        mse_loss_fn = nn.MSELoss()

        prev_entropy = None
        original_param_data = param.data.clone()
        prev_class_loss = float('inf')

        for iteration in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            q_param, entropy, penalty, soft_probs = quant_layer(original_param)

            recon_loss = torch.tensor(0.0, device=device)
            if "weight" in name and activation_input is not None:
                try:
                    conv_layer = next((m for n, m in model.named_modules() if n == layer_name and isinstance(m, nn.Conv2d)), None)
                    if conv_layer:
                        q_out = nn.functional.conv2d(activation_input, q_param,
                                                     stride=conv_layer.stride,
                                                     padding=conv_layer.padding,
                                                     dilation=conv_layer.dilation,
                                                     groups=conv_layer.groups)
                        o_out = nn.functional.conv2d(activation_input, original_param,
                                                     stride=conv_layer.stride,
                                                     padding=conv_layer.padding,
                                                     dilation=conv_layer.dilation,
                                                     groups=conv_layer.groups)
                        recon_loss = mse_loss_fn(q_out, o_out)
                    elif len(original_param.shape) == 2:
                        flat = activation_input.view(activation_input.size(0), -1)
                        q_out = nn.functional.linear(flat, q_param)
                        o_out = nn.functional.linear(flat, original_param)
                        recon_loss = mse_loss_fn(q_out, o_out)
                except Exception as e:
                    print(f"⚠️ Error computing recon for {name}: {e}")
                    recon_loss = torch.tensor(0.0, device=device)

            with torch.no_grad():
                param.data = q_param.detach()
                outputs = model(images)
                class_loss = nn.CrossEntropyLoss()(outputs, labels)
                param.data = original_param_data

            # if class_loss > 0.2:
            #     break
            prev_class_loss = class_loss

            total_loss = 0.1 * recon_loss + 0.9 * class_loss + 0.0 * entropy
            total_loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                delta_entropy = entropy.item() - prev_entropy if prev_entropy is not None else 0.0
                print(f"Iter {iteration:03d}: recon={recon_loss.item():.6f}, "
                      f"class={class_loss.item():.4f}, entropy={entropy.item():.2f}, "
                      f"Δentropy={delta_entropy:.2f}, total_loss={total_loss:.4f}, "
                      f"penalty={penalty.item():.4f}, "
                      f"w_min={quant_layer.w_min.item():.4f}, w_max={quant_layer.w_max.item():.4f}")
                prev_entropy = entropy.item()

        with torch.no_grad():
            layer_soft_probs[name] = soft_probs.detach().cpu()
            final_param, _, _, _ = quant_layer(original_param)
            param.copy_(final_param.to(param.device))

        del quant_layer, optimizer, soft_probs, q_param
        if activation_input is not None:
            del activation_input
        torch.cuda.empty_cache()

        data_iterator = iter(test_loader)
        images, labels = next(data_iterator)
        images, labels = images[:2].to(device), labels[:2].to(device)
        with torch.no_grad():
            model(images)
        print(f"🔁 Refreshed activations after {name}")

    print("\n✅ Full-layer optimization complete.")

# ==== Run ====
print("📊 Accuracy BEFORE quantization:")
evaluate(model, test_loader)

optimize_all_layers(model, test_loader)

print("📊 Accuracy AFTER quantization:")
evaluate(model, test_loader)
