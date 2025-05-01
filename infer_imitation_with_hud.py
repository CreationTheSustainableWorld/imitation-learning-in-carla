import os
import pygame
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import carla
import random
import time

# モデル定義（trainと一致させる！）
class ImitationModel(nn.Module):
    def __init__(self):
        super(ImitationModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 53 * 53, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.mlp(x)


# 初期設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImitationModel().to(device)
model.load_state_dict(torch.load("imitation_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

# 初期化
pygame.init()
display = pygame.display.set_mode((800, 600))  # HUD表示用
clock = pygame.time.Clock()

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
map = world.get_map()

# 車両スポーン
vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
spawn_point = random.choice(map.get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# カメラセンサー
camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

latest_image = None
frame_count = 0
MAX_FRAMES = 300

# カメラ画像取得コールバック
def camera_callback(image):
    global latest_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    latest_image = array

camera.listen(camera_callback)

try:
    while frame_count < MAX_FRAMES:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        if latest_image is None:
            continue

        img_tensor = transform(latest_image).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(img_tensor).squeeze().cpu().numpy()

        steer, throttle, brake = action
        control = carla.VehicleControl(
            steer=float(steer),
            throttle=max(float(throttle), 0.0),
            brake=max(float(brake), 0.0),
            manual_gear_shift=False
        )
        vehicle.apply_control(control)

        # HUD 表示
        surface = pygame.surfarray.make_surface(np.rot90(latest_image))
        display.blit(surface, (0, 0))
        pygame.display.flip()

        frame_count += 1

    print("✅ 推論による走行完了")

finally:
    if camera.is_listening:
        camera.stop()
    if vehicle.is_alive:
        vehicle.destroy()
    pygame.quit()
