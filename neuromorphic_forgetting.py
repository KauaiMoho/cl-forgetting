import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASK_RANGES = [(0, 1), (2, 4), (5, 7), (8, 9)]

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

class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x1 = F.relu(self.bn1(self.fc1(x)))
        x2 = F.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2 + x1)
        x3 = F.relu(self.bn3(self.fc3(x2)))
        return self.fc4(x3)

    def get_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4]


class Hippocampus(nn.Module):
    def __init__(self, num_layers, feature_dim=6):
        super().__init__()
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(num_layers * feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_layers),
            nn.Sigmoid(),
        )

    def forward(self, layer_features):
        x = layer_features.view(1, -1)
        return self.net(x).squeeze(0) * 0.999 + 0.0001

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
        
        # batch_classes = set(labels.cpu().numpy())
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
        fisher_n = normalize_scores(self.fisher)
        mas_n = normalize_scores(self.mas)
        si_n = normalize_scores(self.si)
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

def train_meta_learning(
    mainnet, hippocampus, importance_cache, si_tracker, replay_buffer,
    train_loader, main_optimizer, hippocampus_optimizer, task_id, current_task_classes, epochs=4
):
    num_layers = len(mainnet.get_layers())
    
    fisher = compute_fisher_fast(mainnet, train_loader, current_task_classes)
    mas = compute_mas_fast(mainnet, train_loader)
    si = [si_tracker.layer_score(layer) for layer in mainnet.get_layers()]
    if task_id > 0 and replay_buffer.has_data():
        old_x, old_y, old_c = replay_buffer.get_all_legacy_meta()
        cos = compute_gradient_conflict_fast(mainnet, train_loader, current_task_classes, old_x, old_y, old_c)
    else:
        cos = [torch.tensor(0.0, device=DEVICE)] * num_layers
    importance_cache.update(fisher, mas, si, cos)

    for epoch in range(epochs):
        total_main_loss, total_meta_loss, step_count = 0.0, 0.0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            step_count += 1
                
            hippocampus.train()
            features = importance_cache.build_features(num_layers, task_id)
            plasticities = hippocampus(features)
            
            baseline_loss = 0.0
            has_history = task_id > 0 and replay_buffer.has_data()
            
            if has_history:
                rep_x, rep_y, rep_classes = replay_buffer.sample_worst_task(task_id, n=images.size(0))
                mainnet.eval()
                with torch.no_grad():
                    val_out = mainnet(rep_x)
                    val_mask = get_dynamic_mask(val_out, rep_classes)
                    baseline_loss = F.cross_entropy(val_out + val_mask, rep_y).item()

            mainnet.train()
            main_optimizer.zero_grad()
            out = mainnet(images)
            
            mask_curr = get_dynamic_mask(out, current_task_classes)
            loss = F.cross_entropy(out + mask_curr, labels)
            loss.backward()
            
            with torch.no_grad():
                for i, layer in enumerate(mainnet.get_layers()):
                    scale = plasticities[i].item()
                    for param in layer.parameters():
                        if param.grad is not None:
                            param.grad.mul_(scale)
            
            main_optimizer.step()
            si_tracker.update(mainnet)
            total_main_loss += loss.item()
            
            if has_history:
                mainnet.eval()
                mainnet.zero_grad()
                val_out_old = mainnet(rep_x)
                val_mask_old = get_dynamic_mask(val_out_old, rep_classes)
                loss_old = F.cross_entropy(val_out_old + val_mask_old, rep_y)
                loss_old.backward(retain_graph=True)

                old_grads = []
                for layer in mainnet.get_layers():
                    g = torch.cat([
                        p.grad.detach().flatten()
                        for p in layer.parameters()
                        if p.grad is not None
                    ])
                    old_grads.append(g)

                mainnet.zero_grad()
                mainnet.train()
                mainnet.zero_grad()

                out = mainnet(images)
                mask_curr = get_dynamic_mask(out, current_task_classes)
                loss_new = F.cross_entropy(out + mask_curr, labels)
                loss_new.backward()

                new_grads = []
                for layer in mainnet.get_layers():
                    g = torch.cat([
                        p.grad.detach().flatten()
                        for p in layer.parameters()
                        if p.grad is not None
                    ])
                    new_grads.append(g)

                mainnet.zero_grad()

                meta_signal = 0.0
                for g_old, g_new in zip(old_grads, new_grads):
                    meta_signal += F.cosine_similarity(
                        g_old.unsqueeze(0),
                        g_new.unsqueeze(0)
                    )

                meta_signal = meta_signal / len(old_grads)

                total_meta_loss += meta_signal.item()

                hippocampus_optimizer.zero_grad()

                pseudo_grad = torch.zeros_like(plasticities)

                for i in range(num_layers):
                    pseudo_grad[i] = (1.0 - meta_signal) * (plasticities[i] - plasticities.mean())

                plasticities.backward(gradient=pseudo_grad)

                hippocampus_optimizer.step()
                
        fisher = compute_fisher_fast(mainnet, train_loader, current_task_classes)
        mas = compute_mas_fast(mainnet, train_loader)
        importance_cache.update(fisher, mas, si, cos)
            
        meta_print = f"| Meta Delta: {total_meta_loss/step_count:.6f}" if task_id > 0 else ""
        print(f"Epoch {epoch+1} | Inner Main Loss: {total_main_loss/step_count:.4f} {meta_print}")


def evaluate(model, test_loader, eval_task_classes, eval_task_id):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            
            mask = get_dynamic_mask(out, eval_task_classes)
            predicted = (out + mask).argmax(dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / max(total, 1)
    print(f"Task {eval_task_id + 1} Accuracy: {acc * 100:.2f}%")
    return acc


def get_dataloaders():
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)

    def subset(ds, lo, hi):
        mask = (ds.targets >= lo) & (ds.targets <= hi)
        return Subset(ds, torch.where(mask)[0])

    train_loaders = [DataLoader(subset(mnist_train, lo, hi), batch_size=64, shuffle=True) for lo, hi in TASK_RANGES]
    test_loaders = [DataLoader(subset(mnist_test, lo, hi), batch_size=64, shuffle=False) for lo, hi in TASK_RANGES]
    return train_loaders, test_loaders


if __name__ == "__main__":
    torch.manual_seed(17)
    train_loaders, test_loaders = get_dataloaders()

    mainnet = MainNet().to(DEVICE)
    num_layers = len(mainnet.get_layers())
    hippocampus = Hippocampus(num_layers).to(DEVICE)
    replay_buffer = ReplayBuffer(250)
    si_tracker = SITracker(mainnet)
    importance_cache = ImportanceCache()

    main_optimizer = optim.Adam(mainnet.parameters(), lr=1e-3)
    hippocampus_optimizer = optim.Adam(hippocampus.parameters(), lr=1e-3)

    task_class_registry = {}

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n{'='*50}\nTraining Task {task_id + 1}\n{'='*50}")
        
        current_classes = set()
        for _, lbls in train_loader:
            for l in lbls:
                current_classes.add(l.item())
        task_class_registry[task_id] = current_classes
        
        train_meta_learning(
            mainnet, hippocampus, importance_cache, si_tracker, replay_buffer,
            train_loader, main_optimizer, hippocampus_optimizer, task_id, current_classes, epochs=4
        )
        
        print(f"\nFinal Task {task_id + 1} Accuracy:")
        final_test_acc = evaluate(mainnet, test_loaders[task_id], current_classes, task_id)
        
        with torch.no_grad():
            features = importance_cache.build_features(num_layers, task_id)
            structural_importance = features[:, :3].mean().item() 
            
        vulnerability_score = structural_importance / (1.0 + final_test_acc)
        
        replay_buffer.add_task(task_id, train_loader, vulnerability_score)
        
        print(f"\nPost-Task {task_id + 1} Verification Summary:")
        for prev_id in range(task_id):
            evaluate(mainnet, test_loaders[prev_id], task_class_registry[prev_id], prev_id)