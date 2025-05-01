# enhanced_model.py
import torch.nn as nn
import torch

class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 25 * 25, 128), nn.ReLU(),
        )
        self.steer_head = nn.Sequential(nn.Linear(128, 1), nn.Tanh())       # [-1, 1]
        self.throttle_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid()) # [0, 1]
        self.brake_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())    # [0, 1]

    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        steer = self.steer_head(x)
        throttle = self.throttle_head(x)
        brake = self.brake_head(x)
        return torch.cat([steer, throttle, brake], dim=1)
