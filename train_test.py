import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import data, utils
from models import models
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EWC_LAMBDA = 1

class SITracker:
    def __init__(self, model):
        self.prev_params = {}
        self.si_scores = {}
        for name, p in model.named_parameters():
            self.prev_params[name] = p.detach().clone()
            self.si_scores[name] = torch.zeros_like(p)

    def update(self, model):
        for name, p in model.named_parameters():
            if p.grad is not None:
                delta = p.detach() - self.prev_params[name]
                self.si_scores[name] += (-p.grad.detach() * delta).clamp(min=0)
            self.prev_params[name] = p.detach().clone()

    def layer_score(self, layer):
        scores = []
        for param_name, _ in layer.named_parameters():
            for full_name, score in self.si_scores.items():
                if full_name.endswith(param_name):
                    scores.append(score.mean())
        return torch.stack(scores).mean() if scores else torch.tensor(0.0, device=DEVICE)


class ImportanceCache:
    def __init__(self):
        self.fisher = None
        self.mas = None
        self.si = None

    def update(self, fisher, mas, si):
        self.fisher = fisher
        self.mas = mas
        self.si = si

    def build_features(self, num_layers, task_id):
        fisher_n = utils.normalize_scores(self.fisher)
        mas_n = utils.normalize_scores(self.mas)
        si_n = utils.normalize_scores(self.si)
        layer_pos = torch.tensor(
            [i / max(num_layers - 1, 1) for i in range(num_layers)], device=DEVICE
        )
        task_ctx = torch.full(
            (num_layers,), math.log1p(task_id) / math.log1p(10), device=DEVICE
        )
        return torch.stack([
            torch.stack([fisher_n[i], mas_n[i], si_n[i], layer_pos[i], task_ctx[i]])
            for i in range(num_layers)
        ])

    def combined_importance(self):
        fisher_n = utils.normalize_scores(self.fisher)
        mas_n    = utils.normalize_scores(self.mas)
        si_n     = utils.normalize_scores(self.si)
        return (fisher_n + mas_n + si_n) / 3.0


def train(
    model, hippocampus, importance_cache, si_tracker,
    train_loader, optimizer, hippocampus_optimizer,
    task_id, current_task_classes, param_snapshot, epochs=10
):
    num_layers = len(model.get_layers())

    fisher = utils.compute_fisher_fast(model, train_loader, current_task_classes)
    mas = utils.compute_mas_fast(model, train_loader)
    si = [si_tracker.layer_score(layer) for layer in model.get_layers()]
    importance_cache.update(fisher, mas, si)

    position_prior = torch.tensor(
        [i / max(num_layers - 1, 1) for i in range(num_layers)], device=DEVICE
    )

    for epoch in range(epochs):
        total_main_loss, total_consolidation, step_count = 0.0, 0.0, 0
        epoch_plasticities = []

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            step_count += 1

            hippocampus.train()
            features = importance_cache.build_features(num_layers, task_id)
            plasticities = hippocampus(features)
            epoch_plasticities.append(plasticities.detach().cpu())
            
            model.train()
            optimizer.zero_grad()
            out = model(images)
            mask = utils.get_dynamic_mask(out, current_task_classes)
            task_loss = F.cross_entropy(out + mask, labels)

            consolidation_loss = utils.compute_consolidation_loss(
                model, importance_cache, param_snapshot
            )
            loss = task_loss + EWC_LAMBDA * consolidation_loss
            loss.backward()

            total_main_loss += task_loss.item()
            total_consolidation += consolidation_loss.item()

            layer_signals = []
            for layer in model.get_layers():
                grads = [p.grad.detach().flatten() for p in layer.parameters() if p.grad is not None]
                layer_signals.append(torch.cat(grads).norm() if grads else torch.tensor(0.0, device=DEVICE))
            layer_signals = torch.stack(layer_signals)
            layer_signals = layer_signals / (layer_signals.max() + 1e-8)

            with torch.no_grad():
                for i, layer in enumerate(model.get_layers()):
                    for param in layer.parameters():
                        if param.grad is not None:
                            param.grad.mul_(plasticities[i].item())

            si_tracker.update(model)
            optimizer.step()

            hippocampus_optimizer.zero_grad()
            combined_signal = 0.5 * layer_signals + 0.5 * position_prior
            baseline = combined_signal.mean()
            pseudo_grad = (combined_signal - baseline).detach() * 0.1
            pseudo_grad += 0.05 * (plasticities.detach() - 0.5)
            plasticities.backward(gradient=pseudo_grad)
            hippocampus_optimizer.step()

        fisher = utils.compute_fisher_fast(model, train_loader, current_task_classes)
        mas = utils.compute_mas_fast(model, train_loader)
        importance_cache.update(fisher, mas, si)

        cons_print = f"| Consolidation: {total_consolidation/step_count:.4f}" if param_snapshot is not None else ""

        stacked = torch.stack(epoch_plasticities)
        mean_p = stacked.mean(dim=0)
        std_p = stacked.std(dim=0)
        p_parts = ", ".join(
            f"L{i+1}:{mean_p[i]:.3f}±{std_p[i]:.3f}"
            for i in range(mean_p.shape[0])
        )
        print(f"Epoch {epoch+1} | Loss: {total_main_loss/step_count:.4f} {cons_print} | Plasticities [{p_parts}]")


def evaluate(model, test_loader, eval_task_classes, eval_task_id):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            mask = utils.get_dynamic_mask(out, eval_task_classes)
            predicted = (out + mask).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / max(total, 1)
    print(f"Task {eval_task_id + 1} Accuracy: {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":
    torch.manual_seed(17)
    # train_loaders, test_loaders = data.get_dataloaders_CIFAR100(num_tasks=10, classes_per_task=10)
    # mainnet = models.CIFARNet().to(DEVICE)

    train_loaders, test_loaders = data.get_dataloaders_MNIST([(0, 1), (2, 4), (5, 7), (8, 9)])
    mainnet = models.MNISTNet().to(DEVICE)

    num_layers = len(mainnet.get_layers())
    hippocampus = models.Hippocampus(num_layers).to(DEVICE)
    si_tracker = SITracker(mainnet)
    importance_cache = ImportanceCache()

    optimizer = optim.Adam(mainnet.parameters(), lr=1e-3)
    hippocampus_optimizer = optim.Adam(hippocampus.parameters(), lr=1e-4)

    task_class_registry = {}
    param_snapshot = None

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n{'='*50}\nTraining Task {task_id + 1}\n{'='*50}")

        current_classes = set()
        for _, lbls in train_loader:
            for l in lbls:
                current_classes.add(l.item())
        task_class_registry[task_id] = current_classes

        train(
            mainnet, hippocampus, importance_cache, si_tracker,
            train_loader, optimizer, hippocampus_optimizer,
            task_id, current_classes, param_snapshot, epochs=10
        )

        print(f"\nFinal Task {task_id + 1} Accuracy:")
        evaluate(mainnet, test_loaders[task_id], current_classes, task_id)

        param_snapshot = {}
        for i, layer in enumerate(mainnet.get_layers()):
            for name, param in layer.named_parameters():
                param_snapshot[f"layer{i}_{name}"] = param.detach().clone()

        print(f"\nPost-Task {task_id + 1} Verification Summary:")
        for prev_id in range(task_id):
            evaluate(mainnet, test_loaders[prev_id], task_class_registry[prev_id], prev_id)

# No hippocampus, lambda 1
# Final Task 10 Accuracy:
# Task 10 Accuracy: 47.70%

# Post-Task 10 Verification Summary:
# Task 1 Accuracy: 37.90%
# Task 2 Accuracy: 22.50%
# Task 3 Accuracy: 18.60%
# Task 4 Accuracy: 19.10%
# Task 5 Accuracy: 32.60%
# Task 6 Accuracy: 23.30%
# Task 7 Accuracy: 35.20%
# Task 8 Accuracy: 34.50%
# Task 9 Accuracy: 36.20%

# Hippocampus, lambda 1
# Final Task 10 Accuracy:
# Task 10 Accuracy: 48.00%

# Post-Task 10 Verification Summary:
# Task 1 Accuracy: 46.60%
# Task 2 Accuracy: 26.50%
# Task 3 Accuracy: 28.70%
# Task 4 Accuracy: 23.00%
# Task 5 Accuracy: 34.70%
# Task 6 Accuracy: 24.30%
# Task 7 Accuracy: 37.60%
# Task 8 Accuracy: 38.70%
# Task 9 Accuracy: 38.70%