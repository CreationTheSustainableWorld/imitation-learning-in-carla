import os
import numpy as np
import pygame
import carla
import random
import time
import csv
from tqdm import tqdm

import sys

# PythonAPI/examples を import path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
examples_path = os.path.join(current_dir, '..', 'PythonAPI', 'examples')
sys.path.append(examples_path)

from agents.navigation.behavior_agent import BehaviorAgent

# 保存先
SAVE_PATH_IMG = 'saved_data/images'
SAVE_PATH_ACT = 'saved_data/actions'
CSV_LOG_PATH = 'saved_data/log.csv'

# パラメータ
MAX_FRAMES = 300

# 初期化
pygame.init()
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()

# 車両とカメラの生成
blueprint_library = world.get_blueprint_library()
vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
spawn_point = random.choice(map.get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# エージェント
agent = BehaviorAgent(vehicle, behavior='normal')
destination = random.choice(map.get_spawn_points()).location
agent.set_destination(destination)

# 保存先準備
os.makedirs(SAVE_PATH_IMG, exist_ok=True)
os.makedirs(SAVE_PATH_ACT, exist_ok=True)
csv_file = open(CSV_LOG_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "steer", "throttle", "brake"])

frame_count = 0
progress = tqdm(total=MAX_FRAMES, desc="Collecting Data")

# カメラコールバック
def camera_callback(image):
    global frame_count
    image.convert(carla.ColorConverter.Raw)
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]

    control = vehicle.get_control()
    control_array = np.array([control.steer, control.throttle, control.brake])

    # 保存
    img_filename = f"{frame_count:05d}.png"
    img_path = os.path.join(SAVE_PATH_IMG, img_filename)
    npy_path = os.path.join(SAVE_PATH_ACT, f"{frame_count:05d}.npy")
    image.save_to_disk(img_path)
    np.save(npy_path, control_array)

    # CSVに記録
    csv_writer.writerow([img_filename, *control_array])

    frame_count += 1
    progress.update(1)

    if frame_count >= MAX_FRAMES:
        print("✅ データ収集完了")
        camera.stop()

# カメラスタート
camera.listen(camera_callback)

# メインループ
try:
    while frame_count < MAX_FRAMES:
        if agent.done():
            print("目的地に到達")
            break
        control = agent.run_step()
        control.manual_gear_shift = False
        vehicle.apply_control(control)
        time.sleep(0.05)
finally:
    camera.stop()
    if vehicle.is_alive:
        vehicle.destroy()
    if camera.is_alive:
        camera.destroy()
    csv_file.close()
    progress.close()
    pygame.quit()
