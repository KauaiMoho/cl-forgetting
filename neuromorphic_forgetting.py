import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

#CHECK PERMUTED MNIST
#https://arxiv.org/pdf/1906.00695
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FREEZE_LAMBDA = 0.85

TASK_CLASSES = {
    0: [0, 1],
    1: [2, 3, 4],
    2: [5, 6, 7],
    3: [8, 9]
}

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
            nn.Sigmoid()
        )
        
    def forward(self, layer_features):
        x = layer_features.view(1, -1)
        return self.net(x).squeeze(0) * 0.95 + 0.05

#https://www.pnas.org/doi/10.1073/pnas.1611835114
def compute_isolated_fishers(model, new_images, new_labels, old_images, old_labels, task_id):
    model.zero_grad()
    
    old_out = model(old_images)
    old_mask = torch.full_like(old_out, float('-inf'))
    old_mask[:, TASK_CLASSES[task_id - 1]] = 0.0
    
    old_loss = F.cross_entropy(old_out + old_mask, old_labels)
    old_loss.backward()
    fisher_old = []
    for layer in model.get_layers():
        grads = [p.grad.detach().pow(2).mean() for p in layer.parameters() if p.grad is not None]
        fisher_old.append(torch.stack(grads).mean() if grads else torch.tensor(1e-8, device=DEVICE))
        
    model.zero_grad()
    
    new_out = model(new_images)
    new_mask = torch.full_like(new_out, float('-inf'))
    new_mask[:, TASK_CLASSES[task_id]] = 0.0
    
    new_loss = F.cross_entropy(new_out + new_mask, new_labels)
    new_loss.backward()
    fisher_new = []
    for layer in model.get_layers():
        grads = [p.grad.detach().pow(2).mean() for p in layer.parameters() if p.grad is not None]
        fisher_new.append(torch.stack(grads).mean() if grads else torch.tensor(1e-8, device=DEVICE))
        
    model.zero_grad()
    return fisher_old, fisher_new

#https://www.sciencedirect.com/science/article/pii/S1877050925026316
def compute_mas_per_layer(model, images, task_id):
    model.zero_grad()
    out = model(images)
    out.pow(2).sum().backward()
    scores = []
    for layer in model.get_layers():
        grads = [p.grad.detach().pow(2).mean() for p in layer.parameters() if p.grad is not None]
        scores.append(torch.stack(grads).mean() if grads else torch.tensor(0.0, device=DEVICE))
    model.zero_grad()
    return scores

#https://www.pnas.org/doi/10.1073/pnas.1611835114
def compute_cos_and_fisher_new(model, new_images, new_labels, old_images, old_labels, task_id):
    model.zero_grad()
    old_out = model(old_images)
    old_mask = torch.full_like(old_out, float('-inf'))
    old_mask[:, TASK_CLASSES[task_id - 1]] = 0.0
    F.cross_entropy(old_out + old_mask, old_labels).backward()
    
    old_grads = [torch.cat([p.grad.detach().flatten() for p in layer.parameters() if p.grad is not None]) for layer in model.get_layers()]
    model.zero_grad()

    new_out = model(new_images)
    new_mask = torch.full_like(new_out, float('-inf'))
    new_mask[:, TASK_CLASSES[task_id]] = 0.0
    F.cross_entropy(new_out + new_mask, new_labels).backward()
    
    new_grads, fisher_new = [], []
    for layer in model.get_layers():
        g = torch.cat([p.grad.detach().flatten() for p in layer.parameters() if p.grad is not None])
        new_grads.append(g)
        fisher_new.append(g.pow(2).mean())
    model.zero_grad()

    cos_sims = [F.cosine_similarity(og.unsqueeze(0), ng.unsqueeze(0)).squeeze() for og, ng in zip(old_grads, new_grads)]
    return cos_sims, fisher_new

#https://arxiv.org/abs/1703.04200
class SITracker:
    def __init__(self, model):
        self.prev_params = {}
        self.si_scores   = {}
        for name, p in model.named_parameters():
            self.prev_params[name] = p.detach().clone()
            self.si_scores[name]   = torch.zeros_like(p)

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


def normalize_scores(scores):
    vals = torch.stack([s.detach().clone().float() for s in scores])
    return vals / (1.0 + vals + 1e-8)


def build_layer_features_and_targets(model, new_images, new_labels, old_images, old_labels, si_tracker, task_id):
    layers = model.get_layers()
    num_layers = len(layers)

    fisher_old, fisher_new = compute_isolated_fishers(model, new_images, new_labels, old_images, old_labels, task_id)
    mas_old = compute_mas_per_layer(model, old_images, task_id - 1)
    cos_sim, _ = compute_cos_and_fisher_new(model, new_images, new_labels, old_images, old_labels, task_id)
    si_raw = [si_tracker.layer_score(layer).to(DEVICE) for layer in layers]

    fisher_old_n = normalize_scores(fisher_old)
    mas_old_n = normalize_scores(mas_old)
    si_n = normalize_scores(si_raw)
    cos_n = torch.stack([(c + 1.0) / 2.0 for c in cos_sim])
    layer_pos = torch.tensor([i / max(num_layers - 1, 1) for i in range(num_layers)], device=DEVICE)
    task_context = torch.tensor(task_id / 4.0, device=DEVICE).expand(num_layers)

    features, targets = [], []
    for i in range(num_layers):
        feat = torch.stack([fisher_old_n[i], mas_old_n[i], si_n[i], cos_n[i], layer_pos[i], task_context[i]])
        features.append(feat.detach())

        f_old = fisher_old[i].clamp(min=1e-8)
        f_new = fisher_new[i].clamp(min=1e-8)
        importance = f_old / (f_old + f_new + 1e-8)
        
        plasticity = 1.0 - (FREEZE_LAMBDA * importance)
        targets.append(plasticity.detach())

    return torch.stack(features), torch.stack(targets)


class ReplayBuffer:
    def __init__(self, max_per_task=300):
        self.max_per_task = max_per_task
        self.tasks = {}

    def add_task(self, task_id, dataloader):
        imgs, lbls = [], []
        for x, y in dataloader:
            imgs.append(x); lbls.append(y)
            if sum(t.size(0) for t in imgs) >= self.max_per_task:
                break
        self.tasks[task_id] = (
            torch.cat(imgs)[:self.max_per_task].to(DEVICE),
            torch.cat(lbls)[:self.max_per_task].to(DEVICE),
        )

    def sample(self, n=64):
        if not self.tasks: return None, None
        all_x = torch.cat([x for x, _ in self.tasks.values()])
        all_y = torch.cat([y for _, y in self.tasks.values()])
        idx = torch.randperm(all_x.size(0))[:n]
        return all_x[idx], all_y[idx]

    def has_data(self):
        return bool(self.tasks)


def apply_plasticity_mask(model, plasticities):
    frozen_params, total_params = 0, 0
    for layer, p_score in zip(model.get_layers(), plasticities):
        scale = p_score.item()
        n_params = sum(p.numel() for p in layer.parameters())
        total_params += n_params
        frozen_params += n_params * (1.0 - scale)
        for param in layer.parameters():
            if param.grad is not None:
                param.grad.data.mul_(scale)
    return frozen_params, total_params

def train(mainnet, hippocampus, train_loader, replay_buffer,
          main_optimizer, hippocampus_optimizer, si_tracker):

    criterion = nn.CrossEntropyLoss()
    hippocampus_crit = nn.MSELoss()

    for epoch in range(5):
        total_loss = 0.0
        total_frozen, total_params, batches_masked = 0, 0, 0
        plasticities = torch.zeros(4)

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if task_id > 0 and replay_buffer.has_data():
                old_x, old_y = replay_buffer.sample(n=64)

                feats_tensor, targets_tensor = build_layer_features_and_targets(
                    mainnet, images, labels, old_x, old_y, si_tracker, task_id
                )
                
                hippocampus_optimizer.zero_grad()
                pred = hippocampus(feats_tensor)
                hippocampus_loss = hippocampus_crit(pred, targets_tensor)
                hippocampus_loss.backward()
                hippocampus_optimizer.step()

                hippocampus.eval()
                with torch.no_grad():
                    plasticities = hippocampus(feats_tensor)
                hippocampus.train()

                mainnet.zero_grad()
                main_optimizer.zero_grad()
                
                out = mainnet(images)
                
                mask = torch.full(out.shape, float('-inf')).to(DEVICE)
                unique_labels = labels.unique()
                mask[:, unique_labels] = 0.0
                
                loss = criterion(out + mask, labels)
                loss.backward()
                
                frozen, n_params = apply_plasticity_mask(mainnet, plasticities)
                main_optimizer.step()

                total_frozen += frozen
                total_params += n_params
                batches_masked += 1
            else:
                mainnet.zero_grad()
                main_optimizer.zero_grad()
                out = mainnet(images)
                
                mask = torch.full(out.shape, float('-inf')).to(DEVICE)
                unique_labels = labels.unique()
                mask[:, unique_labels] = 0.0

                loss = criterion(out + mask, labels)
                loss.backward()
                main_optimizer.step()

            si_tracker.update(mainnet)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if batches_masked > 0:
            avg_frozen = total_frozen / batches_masked
            avg_total = total_params / batches_masked
            pct = 100.0 * avg_frozen / avg_total
            layer_info = "  |  ".join(f"L{i+1}: {p.item():.3f}" for i, p in enumerate(plasticities))
            print(f"  Epoch {epoch+1}, Loss: {avg_loss:.4f}  |  "
                  f"Frozen: {int(avg_frozen)}/{int(avg_total)} ({pct:.1f}%)  |  "
                  f"Plasticity [{layer_info}]")
        else:
            print(f"  Epoch {epoch+1}, Loss: {avg_loss:.4f}")


def evaluate(model, test_loader, task_id, task_label):

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            out = model(images)

            all_classes_so_far = []
            for t_id in range(task_id + 1):
                all_classes_so_far.extend(TASK_CLASSES[t_id])
                
            mask = torch.full(out.shape, float('-inf')).to(DEVICE)
            mask[:, all_classes_so_far] = 0.0
            
            predicted = (out + mask).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"  Task {task_label} Accuracy: {acc:.2f}%")
    return acc


def get_dataloaders():
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    ranges = [(0, 1), (2, 4), (4, 8), (8, 9)]

    def subset(ds, r):
        lo, hi = r
        mask = (ds.targets >= lo) & (ds.targets <= hi)
        return Subset(ds, torch.where(mask)[0])

    train_loaders = [DataLoader(subset(mnist_train, r), batch_size=64, shuffle=True) for r in ranges]
    test_loaders = [DataLoader(subset(mnist_test, r), batch_size=64, shuffle=False) for r in ranges]
    return train_loaders, test_loaders


if __name__ == "__main__":
    torch.manual_seed(42)
    train_loaders, test_loaders = get_dataloaders()

    mainnet = MainNet().to(DEVICE)
    num_layers = len(mainnet.get_layers())
    hippocampus = Hippocampus(num_layers).to(DEVICE)
    replay_buffer = ReplayBuffer(300)
    si_tracker = SITracker(mainnet)

    main_optimizer = optim.Adam(mainnet.parameters(), lr=1e-3)
    hippocampus_optimizer = optim.Adam(hippocampus.parameters(), lr=1e-3)

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n{'='*50}\nTraining Task {task_id}\n{'='*50}")
        train(mainnet, hippocampus, train_loader, replay_buffer, main_optimizer, hippocampus_optimizer, si_tracker)
        replay_buffer.add_task(task_id, train_loader)

        print(f"\nEvaluation after Task {task_id + 1}:")
        for prev_id in range(task_id):
            evaluate(mainnet, test_loaders[prev_id], prev_id, prev_id + 1)