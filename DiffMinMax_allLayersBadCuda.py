import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import resnet  # Ensure this is your custom or timm-compatible ResNet18

# ==== Hyperparameters ====
BATCH_SIZE = 128
NUM_BITS = 8
FIXED_T = 100.5
LR = 0.001
NUM_ITERATIONS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Load ResNet ====
model = resnet.resnet18(pretrained=False, device=device)
model.to(device)
state_dict = torch.load('/content/resnet18.pt', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

# ==== Evaluation ====
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ==== Activation Hook ====
temp_activations = {}
def activation_hook(layer_name):
    def hook(module, input, output):
        temp_activations[layer_name] = input[0].detach()
    return hook

# Register hooks for Conv2d and Linear layers
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.register_forward_hook(activation_hook(name))

# ==== Quantizer ====
class MinMaxQuantization(nn.Module):
    def __init__(self, weight, num_levels=2**NUM_BITS, fixed_T=FIXED_T):
        super().__init__()
        self.num_levels = num_levels
        self.fixed_T = fixed_T
        w_min_init = weight.min().detach()
        w_max_init = weight.max().detach()
        range_padding = 0.05 * (w_max_init - w_min_init)
        self.w_min = nn.Parameter(w_min_init - range_padding)
        self.w_max = nn.Parameter(w_max_init + range_padding)

    def forward(self, w):
        EPSILON = 1e-6
        w_min_clamped = self.w_min.clamp(max=self.w_max.item() - EPSILON)
        w_max_clamped = self.w_max.clamp(min=w_min_clamped.item() + EPSILON)
        w_normalized = (w - w_min_clamped) / (w_max_clamped - w_min_clamped + EPSILON)
        q_levels = torch.linspace(0, 1, self.num_levels, device=w.device)
        distances = -torch.abs(w_normalized.unsqueeze(-1) - q_levels)
        soft_weights = torch.softmax(distances * self.fixed_T, dim=-1)
        w_quantized = (soft_weights * q_levels).sum(dim=-1)
        w_dequantized = w_quantized * (w_max_clamped - w_min_clamped) + w_min_clamped
        return w_dequantized

# ==== Full Layer-Wise Optimization ====
def optimize_all_layers(model, test_loader, num_iterations=NUM_ITERATIONS, lr=LR):
    model.eval()
    updated_state_dict = model.state_dict()

    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        model(images)
        outputs = model(images)
        initial_loss = nn.CrossEntropyLoss()(outputs, labels).item()
    print(f"Initial Classification Loss Before Optimization: {initial_loss:.6f}")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        print(f"\nQuantizing {name}...")
        layer_name = name.replace(".weight", "").replace(".bias", "")
        activation_input = temp_activations.get(layer_name, None)
        original_param = param.clone().detach()
        quant_layer = MinMaxQuantization(original_param).to(device)
        optimizer = optim.Adam(quant_layer.parameters(), lr=lr)
        prev_class_loss = float('inf')
        mse_loss_fn = nn.MSELoss()

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            quantized_param = quant_layer(original_param)

            if "weight" in name and activation_input is not None:
                recon_loss = torch.tensor(0.0, device=device)

                if len(original_param.shape) == 4:  # Conv2d
                    conv_layer = None
                    for module_name, module in model.named_modules():
                        if module_name == layer_name and isinstance(module, nn.Conv2d):
                            conv_layer = module
                            break
                    if conv_layer is None:
                        print(f"Skipping {layer_name}: Conv2d layer not found.")
                        continue

                    stride = conv_layer.stride
                    padding = conv_layer.padding
                    dilation = conv_layer.dilation
                    groups = conv_layer.groups

                    quantized_output = nn.functional.conv2d(activation_input, quantized_param, bias=None,
                                                            stride=stride, padding=padding, dilation=dilation, groups=groups)
                    original_output = nn.functional.conv2d(activation_input, original_param, bias=None,
                                                           stride=stride, padding=padding, dilation=dilation, groups=groups)
                    recon_loss = mse_loss_fn(quantized_output, original_output)

                elif len(original_param.shape) == 2:  # Linear
                    act_flat = activation_input.view(activation_input.size(0), -1)
                    quantized_output = nn.functional.linear(act_flat, quantized_param)
                    original_output = nn.functional.linear(act_flat, original_param)
                    recon_loss = mse_loss_fn(quantized_output, original_output)

            else:
                recon_loss = torch.tensor(0.0, device=device)

            class_loss = nn.CrossEntropyLoss()(model(images), labels)
            if class_loss > prev_class_loss:
                print(f"Early stop at iter {iteration}: class_loss increased.")
                break
            prev_class_loss = class_loss

            total_loss = 0.1 * recon_loss + 0.9 * class_loss
            total_loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"Iter {iteration}: recon_loss={recon_loss.item():.8f}, "
                      f"class_loss={class_loss.item():.4f}, "
                      f"w_min={quant_layer.w_min.item():.4f}, "
                      f"w_max={quant_layer.w_max.item():.4f}")

        updated_state_dict[name] = quant_layer(original_param).detach()

    model.load_state_dict(updated_state_dict)
    print("\nFull model quantization complete.")

# ==== Run ====
print("\nAccuracy BEFORE Quantization:")
evaluate(model, test_loader)

optimize_all_layers(model, test_loader)

print("\nAccuracy AFTER Quantization:")
evaluate(model, test_loader)
