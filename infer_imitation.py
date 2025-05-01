import os
import numpy as np
import torch
from PIL import Image
import carla
import pygame
import time
import torchvision.transforms as T
from train_imitation import SimpleCNN

# モデル読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("imitation_model.pth", map_location=device))
model.eval()

# CARLA初期化
pygame.init()
display = pygame.display.set_mode((224, 224))  # ★ ここが可視化のポイント
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.*")[0]
spawn_point = map.get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# カメラ初期化
camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# 推論用前処理
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

frame_count = 0
MAX_FRAMES = 300

def predict(image):
    image = Image.fromarray(image)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor).cpu().numpy()[0]
    return output

# カメラコールバックで推論＋表示＋運転
def camera_callback(image):
    global frame_count
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    
    # 推論
    action = predict(array)
    steer, throttle, brake = action
    control = carla.VehicleControl(
        steer=float(steer),
        throttle=float(np.clip(throttle, 0, 1)),
        brake=float(np.clip(brake, 0, 1)),
        manual_gear_shift=False
    )
    vehicle.apply_control(control)

    # ★ 可視化
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

    frame_count += 1
    if frame_count >= MAX_FRAMES:
        print("✅ 推論による走行完了")
        camera.stop()
        vehicle.destroy()

# 実行
camera.listen(camera_callback)

try:
    while frame_count < MAX_FRAMES:
        time.sleep(0.05)
finally:
    if camera.is_listening:
        camera.stop()
    if vehicle.is_alive:
        vehicle.destroy()
    pygame.quit()
