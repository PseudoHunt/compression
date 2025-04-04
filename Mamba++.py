def compute_wanda_plus_plus_mask(model, dataloader, target_layers, device, topk_ratio=0.5):
    model.eval()
    model.to(device)
    importance_scores = {}
    input_cache = {}

    def save_input(name):
        def hook(module, input, output):
            input_cache[name] = input[0].detach()
        return hook

    handles = []
    for name, layer in target_layers.items():
        layer.weight.requires_grad = True
        layer.weight.retain_grad()
        handles.append(layer.register_forward_hook(save_input(name)))

    # Run forward & backward pass on one batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    model.zero_grad()
    output = model(**batch).logits
    pseudo_labels = output.detach()
    loss = torch.nn.MSELoss()(output, pseudo_labels)
    loss.backward()

    for name, layer in target_layers.items():
        weight = layer.weight.data
        grad = layer.weight.grad
        inp = input_cache[name]  # [B, T, D]
        inp_norm = torch.norm(inp, dim=1).mean(dim=0, keepdim=True)  # [1, D]
        score = torch.abs(weight) * inp_norm.T * torch.abs(grad)
        flat = score.view(-1)
        threshold = torch.kthvalue(flat, int((1 - topk_ratio) * flat.numel())).values
        mask = (score >= threshold).float()
        importance_scores[name] = mask

    for h in handles:
        h.remove()

    return importance_scores
