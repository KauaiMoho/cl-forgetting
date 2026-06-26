import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#In the future try out a task classifier model to identify task boundaries
def get_dynamic_mask(logits, allowed_classes):
    mask = torch.full(logits.shape, float('-inf'), device=DEVICE)
    if not allowed_classes:
        return torch.zeros_like(logits)
    for c in allowed_classes:
        mask[:, c] = 0.0
    return mask

def normalize_scores(scores):
    vals = torch.stack([s.detach().clone().float() for s in scores])
    return vals / (1.0 + vals + 1e-8)

def compute_fisher_fast(model, loader, task_classes, max_batches=2):
    model.eval()
    layer_accum = [0.0] * len(model.get_layers())
    n_batches = 0
    for images, labels in loader:
        if n_batches >= max_batches:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        out = model(images)
        
        mask = get_dynamic_mask(out, task_classes)
        F.cross_entropy(out + mask, labels).backward()
        for i, layer in enumerate(model.get_layers()):
            grads = [p.grad.detach().pow(2).mean() for p in layer.parameters() if p.grad is not None]
            if grads:
                layer_accum[i] += torch.stack(grads).mean().item()
        n_batches += 1
    model.zero_grad()
    return [torch.tensor(s / max(n_batches, 1), device=DEVICE) for s in layer_accum] # torch.zeros(len(layer_accum), device=DEVICE)


def compute_mas_fast(model, loader, max_batches=2):
    model.eval()
    layer_accum = [0.0] * len(model.get_layers())
    n_batches = 0
    for images, _ in loader:
        if n_batches >= max_batches:
            break
        images = images.to(DEVICE)
        model.zero_grad()
        out = model(images)
        out.pow(2).sum().backward()
        for i, layer in enumerate(model.get_layers()):
            grads = [p.grad.detach().pow(2).mean() for p in layer.parameters() if p.grad is not None]
            if grads:
                layer_accum[i] += torch.stack(grads).mean().item()
        n_batches += 1
    model.zero_grad()
    return [torch.tensor(s / max(n_batches, 1), device=DEVICE) for s in layer_accum]

def compute_consolidation_loss(model, importance_cache, param_snapshot):
    if param_snapshot is None:
        return torch.tensor(0.0, device=DEVICE)

    importance = importance_cache.combined_importance()
    layers = model.get_layers()
    total_loss = torch.tensor(0.0, device=DEVICE)

    for i, layer in enumerate(layers):
        layer_importance = importance[i]
        for name, param in layer.named_parameters():
            old_key = f"layer{i}_{name}"
            if old_key not in param_snapshot:
                continue
            delta = (param - param_snapshot[old_key]).pow(2).sum()
            total_loss = total_loss + layer_importance * delta

    return total_loss