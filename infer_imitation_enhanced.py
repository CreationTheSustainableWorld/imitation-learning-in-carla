import os
import numpy as np
import torch
from PIL import Image
import carla
import pygame
import time
import csv
import torchvision.transforms as T
from enhanced_model import EnhancedCNN  # モデル定義ファイル

# モデル読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCNN().to(device)
model.load_state_dict(torch.load("enhanced_imitation_model.pth", map_location=device))
model.eval()

# 保存先
LOG_DIR = "logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
csv_path = os.path.join(LOG_DIR, "log.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "steer", "throttle", "brake"])

# CARLA初期化
pygame.init()
display = pygame.display.set_mode((224, 224))
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.*")[0]
spawn_point = map.get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# 推論前処理
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

# カメラコールバック
def camera_callback(image):
    global frame_count
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]

    # 推論 & 制御
    steer, throttle, brake = predict(array)
    control = carla.VehicleControl(
        steer=float(np.clip(steer, -1, 1)),
        throttle=float(np.clip(throttle, 0, 1)),
        brake=float(np.clip(brake, 0, 1)),
        manual_gear_shift=False
    )
    vehicle.apply_control(control)

    # ログ保存
    img_filename = f"{frame_count:05d}.png"
    Image.fromarray(array).save(os.path.join(IMG_DIR, img_filename))
    csv_writer.writerow([img_filename, steer, throttle, brake])

    # 可視化
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

    frame_count += 1
    if frame_count >= MAX_FRAMES:
        print("✅ 推論走行完了")
        camera.stop()

# 実行開始
camera.listen(camera_callback)

try:
    while frame_count < MAX_FRAMES:
        time.sleep(0.05)
finally:
    if camera.is_listening:
        camera.stop()
    if vehicle.is_alive:
        vehicle.destroy()
    if camera.is_alive:
        camera.destroy()
    csv_file.close()
    pygame.quit()
