import os
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ✅ デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセット定義
class CarlaDataset(Dataset):
    def __init__(self, img_dir, act_dir, transform=None):
        self.img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
        self.act_paths = sorted(glob(os.path.join(act_dir, "*.npy")))
        self.transform = transform or T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        act = np.load(self.act_paths[idx])
        return img, torch.tensor(act, dtype=torch.float32)

# ✅ 改良モデル定義
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

# 学習用コード
def train():
    dataset = CarlaDataset("saved_data/images", "saved_data/actions",
                           transform=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EnhancedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(10):
        total_loss = 0
        steer_loss = 0
        throttle_loss = 0
        brake_loss = 0

        for images, actions in loader:
            images = images.to(device)
            actions = actions.to(device)

            preds = model(images)
            loss = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steer_loss += nn.functional.mse_loss(preds[:, 0], actions[:, 0]).item()
            throttle_loss += nn.functional.mse_loss(preds[:, 1], actions[:, 1]).item()
            brake_loss += nn.functional.mse_loss(preds[:, 2], actions[:, 2]).item()

        avg_batches = len(loader)
        print(f"Epoch {epoch+1}: Total={total_loss/avg_batches:.4f} | "
              f"Steer={steer_loss/avg_batches:.4f} | "
              f"Throttle={throttle_loss/avg_batches:.4f} | "
              f"Brake={brake_loss/avg_batches:.4f}")

    torch.save(model.state_dict(), "enhanced_imitation_model.pth")
    print("✅ 学習完了 & 強化モデル保存")

if __name__ == "__main__":
    train()
