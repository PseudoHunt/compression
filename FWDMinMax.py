import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import timm

# Hyperparameters
BATCH_SIZE = 128
NUM_BITS = 4
FIXED_T = 100.5
LR = 0.001
NUM_ITERATIONS = 100

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

temp_activations = {}

def activation_hook(layer_name):
    def hook(module, input, output):
        temp_activations[layer_name] = input[0].detach()
    return hook

for name, layer in model.named_modules():
    if isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(activation_hook(name))

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

def forward_gradient_update_minmax(quant_layer, original_weight, activation_input, recon_loss_fn, model, images, labels, loss_weight=0.1, class_weight=0.9, lr=0.01, epsilon=1e-3):
    w_min = quant_layer.w_min.detach().clone()
    w_max = quant_layer.w_max.detach().clone()
    v = torch.randn(2).to(original_weight.device)
    v = v / v.norm()

    w_min_pert = w_min + epsilon * v[0]
    w_max_pert = w_max + epsilon * v[1]

    quant_layer.w_min.data = w_min_pert
    quant_layer.w_max.data = w_max_pert
    quantized_weight_pert = quant_layer(original_weight)

    quantized_output_pert = F.conv2d(activation_input, quantized_weight_pert, stride=original_weight.shape[2], padding=original_weight.shape[3])
    original_output = F.conv2d(activation_input, original_weight, stride=original_weight.shape[2], padding=original_weight.shape[3])
    recon_loss_pert = recon_loss_fn(quantized_output_pert, original_output)

    class_loss_pert = F.cross_entropy(model(images), labels)
    loss_pert = loss_weight * recon_loss_pert + class_weight * class_loss_pert

    quant_layer.w_min.data = w_min
    quant_layer.w_max.data = w_max
    quantized_weight_base = quant_layer(original_weight)

    quantized_output_base = F.conv2d(activation_input, quantized_weight_base, stride=original_weight.shape[2], padding=original_weight.shape[3])
    recon_loss_base = recon_loss_fn(quantized_output_base, original_output)

    class_loss_base = F.cross_entropy(model(images), labels)
    loss_base = loss_weight * recon_loss_base + class_weight * class_loss_base

    grad_est = (loss_pert - loss_base) / epsilon * v

    quant_layer.w_min.data -= lr * grad_est[0]
    quant_layer.w_max.data -= lr * grad_est[1]

    return {
        'loss_pert': loss_pert.item(),
        'loss_base': loss_base.item(),
        'grad_est': grad_est.detach().cpu().numpy(),
        'w_min': quant_layer.w_min.item(),
        'w_max': quant_layer.w_max.item()
    }

def optimize_per_layer(model, test_loader, num_iterations=NUM_ITERATIONS, lr=LR):
    model.eval()
    updated_state_dict = model.state_dict()
    quantization_layers = {}

    print("\nStarting per-layer quantization optimization...")

    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        model(images)

    with torch.no_grad():
        outputs = model(images)
        initial_loss = nn.CrossEntropyLoss()(outputs, labels).item()
    print(f"Initial Classification Loss Before Optimization: {initial_loss:.6f}")

    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            print(f"\nOptimizing {name}...")
            layer_name = name.replace(".weight", "")

            if layer_name not in temp_activations:
                print(f"Skipping {layer_name}: No activation found.")
                continue

            original_weight = param.clone().detach()
            quant_layer = MinMaxQuantization(original_weight).to(device)
            mse_loss_fn = nn.MSELoss()
            activation_input = temp_activations[layer_name]

            prev_class_loss = float('inf')
            for iteration in range(num_iterations):
                fg_result = forward_gradient_update_minmax(
                    quant_layer, original_weight, activation_input,
                    mse_loss_fn, model, images, labels,
                    loss_weight=0.1, class_weight=0.9, lr=lr
                )

                if fg_result['loss_pert'] > prev_class_loss:
                    print(f"Early stop at iter {iteration}: class_loss increased.")
                    break
                prev_class_loss = fg_result['loss_pert']

                if iteration % 10 == 0:
                    print(f"Iter {iteration}: fg_loss={fg_result['loss_pert']:.4f}, "
                          f"w_min={fg_result['w_min']:.4f}, w_max={fg_result['w_max']:.4f}")

            updated_state_dict[name] = quant_layer(original_weight).detach()

    model.load_state_dict(updated_state_dict)
    print("\nPer-layer optimization complete.")

evaluate(model, test_loader)  # Before quantization
optimize_per_layer(model, test_loader)
evaluate(model, test_loader)  # After quantization
