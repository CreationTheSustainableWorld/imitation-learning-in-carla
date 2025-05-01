import cv2
import os
from glob import glob

image_dir = "logs/images"
output_path = "logs/output_video.avi"
fps = 10

img_files = sorted(glob(os.path.join(image_dir, "*.png")))
if not img_files:
    raise RuntimeError("❌ 画像ファイルが見つかりませんでした")

frame = cv2.imread(img_files[0])
height, width, _ = frame.shape
size = (width, height)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
for fname in img_files:
    frame = cv2.imread(fname)
    out.write(frame)
out.release()
print(f"✅ 動画保存完了: {output_path}")
