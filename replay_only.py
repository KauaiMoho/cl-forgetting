import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import data, utils
from models import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_RANGES = [(0, 1), (2, 4), (5, 7), (8, 9)]

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
        self.cos = None

    def update(self, fisher, mas, si, cos):
        self.fisher = fisher
        self.mas = mas
        self.si = si
        self.cos = cos

    def build_features(self, num_layers, task_id):
        fisher_n = utils.normalize_scores(self.fisher)
        mas_n = utils.normalize_scores(self.mas)
        si_n = utils.normalize_scores(self.si)
        cos_n = torch.stack([(c + 1.0) / 2.0 for c in self.cos])
        layer_pos = torch.tensor([i / max(num_layers - 1, 1) for i in range(num_layers)], device=DEVICE)
        task_ctx = torch.tensor(task_id / max(task_id, 1), device=DEVICE).expand(num_layers)

        features = []
        for i in range(num_layers):
            feat = torch.stack([fisher_n[i], mas_n[i], si_n[i], cos_n[i], layer_pos[i], task_ctx[i]])
            features.append(feat.detach())
        return torch.stack(features)


class ReplayBuffer:
    def __init__(self, max_per_task=250):
        self.max_per_task = max_per_task
        self.tasks = {}
        self.task_classes = {}
        self.vulnerability_scores = {}
        self.insertion_order = []

    def add_task(self, task_id, dataloader, vulnerability_score):
        imgs, lbls = [], []
        discovered = set()
        for x, y in dataloader:
            imgs.append(x)
            lbls.append(y)
            for label in y:
                discovered.add(label.item())
            if sum(t.size(0) for t in imgs) >= self.max_per_task:
                break
        if imgs:
            self.tasks[task_id] = (
                torch.cat(imgs)[:self.max_per_task].to(DEVICE),
                torch.cat(lbls)[:self.max_per_task].to(DEVICE),
            )
            self.task_classes[task_id] = discovered
            self.vulnerability_scores[task_id] = float(vulnerability_score)
            if task_id not in self.insertion_order:
                self.insertion_order.append(task_id)

    def sample_worst_task(self, current_task_id, n=48):
        if not self.tasks:
            return None, None, None
        
        past_task_ids = list(self.tasks.keys())
        priority_logits = []
        
        for t_id in past_task_ids:
            age = max(1, current_task_id - t_id)
            init_vulnerability = self.vulnerability_scores[t_id]
            
            priority = init_vulnerability * (1.0 + 0.5 * age)
            priority_logits.append(priority)
            
        logits_tensor = torch.tensor(priority_logits, device=DEVICE)
        probabilities = F.softmax(logits_tensor, dim=0)
        
        chosen_idx = torch.multinomial(probabilities, 1).item()
        chosen_task = past_task_ids[chosen_idx]
        
        task_imgs, task_lbls = self.tasks[chosen_task]
        ty = self.task_classes[chosen_task]
        
        final_idx = torch.randperm(task_imgs.size(0))[:n]
        return task_imgs[final_idx], task_lbls[final_idx], ty

    def has_data(self):
        return bool(self.tasks)

    def get_all_legacy_meta(self):
        if not self.tasks:
            return None, None, set()
        all_x = torch.cat([t[0] for t in self.tasks.values()])
        all_y = torch.cat([t[1] for t in self.tasks.values()])
        all_classes = set()
        for c_set in self.task_classes.values():
            all_classes.update(c_set)
        return all_x, all_y, all_classes
    
def train_replay_only(model, replay_buffer, train_loader, main_optimizer, task_id, current_task_classes, epochs=4):
    for epoch in range(epochs):
        total_loss, step_count = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            step_count += 1

            model.train()
            main_optimizer.zero_grad()
            out = model(images)
            mask_curr = utils.get_dynamic_mask(out, current_task_classes)
            loss = F.cross_entropy(out + mask_curr, labels)

            if task_id > 0 and replay_buffer.has_data():
                rep_x, rep_y, rep_classes = replay_buffer.sample_worst_task(task_id, n=images.size(0))
                rep_out = model(rep_x)
                rep_mask = utils.get_dynamic_mask(rep_out, rep_classes)
                loss_replay = F.cross_entropy(rep_out + rep_mask, rep_y)
                loss = 0.5 * loss + 0.5 * loss_replay  # equal weighting

            loss.backward()
            main_optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss/step_count:.4f}")

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
    # train_loaders, test_loaders = data.get_dataloaders_MNIST(TASK_RANGES)
    train_loaders, test_loaders = data.get_dataloaders_CIFAR100(num_tasks=10, classes_per_task=10)

    mainnet = models.SimpleCNN().to(DEVICE)
    num_layers = len(mainnet.get_layers())
    replay_buffer = ReplayBuffer(200)
    si_tracker = SITracker(mainnet)
    importance_cache = ImportanceCache()

    main_optimizer = optim.Adam(mainnet.parameters(), lr=1e-3)
    
    task_class_registry = {}

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n{'='*50}\nTraining Task {task_id + 1}\n{'='*50}")

        current_classes = set()
        for _, lbls in train_loader:
            for l in lbls:
                current_classes.add(l.item())
        task_class_registry[task_id] = current_classes

        train_replay_only(
            mainnet, replay_buffer, train_loader,
            main_optimizer, task_id, current_classes, epochs=10
        )

        print(f"\nFinal Task {task_id + 1} Accuracy:")
        final_test_acc = evaluate(mainnet, test_loaders[task_id], current_classes, task_id)

        structural_importance = 0.5  # neutral fallback for replay_only

        vulnerability_score = structural_importance / (1.0 + final_test_acc)
        replay_buffer.add_task(task_id, train_loader, vulnerability_score)

        print(f"\nPost-Task {task_id + 1} Verification Summary:")
        for prev_id in range(task_id):
            evaluate(mainnet, test_loaders[prev_id], task_class_registry[prev_id], prev_id)
