import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
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
    
class CIFARNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32 -> 16
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 16 -> 8
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    def get_layers(self):
        return [self.conv1, self.conv2, self.conv3, self.fc]


class Hippocampus(nn.Module):
    def __init__(self, num_layers, feature_dim=5):
        super().__init__()
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(num_layers * feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_layers),
        )

    def forward(self, layer_features):
        x = layer_features.view(1, -1)
        raw = self.net(x).squeeze(0)
        return 0.05 + 0.95 * torch.sigmoid(raw)