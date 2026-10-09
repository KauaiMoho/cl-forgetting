import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import data, utils
from models import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EWC_LAMBDA = 0

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

    def combined_importance(self):
        fisher_n = utils.normalize_scores(self.fisher)
        mas_n    = utils.normalize_scores(self.mas)
        si_n     = utils.normalize_scores(self.si)
        return (fisher_n + mas_n + si_n) / 3.0

def train(
    model, importance_cache, si_tracker,
    train_loader, optimizer,
    task_id, current_task_classes, param_snapshot, epochs=10
):
    num_layers = len(model.get_layers())

    fisher = utils.compute_fisher_fast(model, train_loader, current_task_classes)
    mas = utils.compute_mas_fast(model, train_loader)
    si = [si_tracker.layer_score(layer) for layer in model.get_layers()]
    importance_cache.update(fisher, mas, si)

    for epoch in range(epochs):
        total_main_loss, total_consolidation, step_count = 0.0, 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            step_count += 1

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

            si_tracker.update(model)
            optimizer.step()

        fisher = utils.compute_fisher_fast(model, train_loader, current_task_classes)
        mas = utils.compute_mas_fast(model, train_loader)
        importance_cache.update(fisher, mas, si)

        cons_print = f"| Consolidation: {total_consolidation/step_count:.4f}" if param_snapshot is not None else ""
        print(f"Epoch {epoch+1} | Loss: {total_main_loss/step_count:.4f} {cons_print}")


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
    si_tracker = SITracker(mainnet)
    importance_cache = ImportanceCache()
    optimizer = optim.Adam(mainnet.parameters(), lr=1e-3)

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
            mainnet, importance_cache, si_tracker,
            train_loader, optimizer,
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