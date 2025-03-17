import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import timm

# Hyperparameters
BATCH_SIZE = 128
NUM_BITS = 3
FIXED_T = 100.5  # Fixed temperature for soft rounding
LR = 0.001  # Learning rate
NUM_ITERATIONS = 100  # Per-layer optimization iterations

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Pretrained ResNet18 from timm
import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18.resnet18(pretrained=False, device=device)
model.to(device)

state_dict = torch.load('/content/resnet18.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

print("\nAccuracy BEFORE Quantization:")

# Function to evaluate model accuracy
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

# ====== Hook for capturing activations ======
temp_activations = {}

def activation_hook(layer_name):
    def hook(module, input, output):
        temp_activations[layer_name] = input[0].detach()
    return hook

# Register hooks for Conv2D layers
for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(activation_hook(name))

# ====== Differentiable Min-Max Quantization Module ======
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

# ====== Per-Layer Optimization with Activation Usage ======
def optimize_per_layer(model, test_loader, num_iterations=NUM_ITERATIONS, lr=LR):
    model.eval()
    updated_state_dict = model.state_dict()
    quantization_layers = {}

    print("\nStarting per-layer quantization optimization...")

    # Get one batch for activations and loss computation
    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images, labels = images.to(device), labels.to(device)

    # Capture activations
    with torch.no_grad():
        model(images)

    # Initial accuracy reference
    with torch.no_grad():
        outputs = model(images)
        initial_loss = nn.CrossEntropyLoss()(outputs, labels).item()
    print(f"Initial Classification Loss Before Optimization: {initial_loss:.6f}")

    # Optimize layer by layer
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            print(f"\nOptimizing {name}...")
            layer_name = name.replace(".weight", "")

            if layer_name not in temp_activations:
                print(f"Skipping {layer_name}: No activation found.")
                continue

            original_weight = param.clone().detach()
            quant_layer = MinMaxQuantization(original_weight).to(device)
            optimizer = optim.Adam(quant_layer.parameters(), lr=lr)
            mse_loss_fn = nn.MSELoss()
            activation_input = temp_activations[layer_name]  # Correct input for this layer

            prev_class_loss = float('inf')
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                quantized_weight = quant_layer(original_weight)

                quantized_output = nn.functional.conv2d(
                    activation_input, quantized_weight, stride=param.shape[2], padding=param.shape[3]
                )
                original_output = nn.functional.conv2d(
                    activation_input, original_weight, stride=param.shape[2], padding=param.shape[3]
                )

                recon_loss = mse_loss_fn(quantized_output, original_output)
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

            updated_state_dict[name] = quant_layer(original_weight).detach()

    model.load_state_dict(updated_state_dict)
    print("\nPer-layer optimization complete.")

# ====== Run Optimization and Evaluation ======
evaluate(model, test_loader)  # Before quantization
optimize_per_layer(model, test_loader)
evaluate(model, test_loader)  # After quantization
