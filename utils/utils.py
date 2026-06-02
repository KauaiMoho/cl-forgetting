import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return [torch.tensor(s / max(n_batches, 1), device=DEVICE) for s in layer_accum]


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


def compute_gradient_conflict_fast(model, loader, current_task_classes, old_images, old_labels, old_classes, max_batches=2):
    model.eval()
    layers = model.get_layers()
    model.zero_grad()
    old_out = model(old_images)
    mask_old = get_dynamic_mask(old_out, old_classes)
    F.cross_entropy(old_out + mask_old, old_labels).backward()
    old_grads = [
        torch.cat([p.grad.detach().flatten() for p in layer.parameters() if p.grad is not None]) for layer in layers
    ]
    model.zero_grad()

    new_grad_accum = [torch.zeros_like(g) for g in old_grads]
    n_batches = 0
    for images, labels in loader:
        if n_batches >= max_batches:
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        out = model(images)
        # batch_classes = set(labels.cpu().numpy())
        mask_new = get_dynamic_mask(out, current_task_classes)
        F.cross_entropy(out + mask_new, labels).backward()
        for i, layer in enumerate(layers):
            g = torch.cat([p.grad.detach().flatten() for p in layer.parameters() if p.grad is not None])
            new_grad_accum[i] += g
        n_batches += 1
    model.zero_grad()
    return [F.cosine_similarity(og.unsqueeze(0), (ng / max(n_batches, 1)).unsqueeze(0)).squeeze() 
            for og, ng in zip(old_grads, new_grad_accum)]