from ultralytics import YOLO

# 导入模型
model = YOLO("/root/workspace/cloudcare-flask-app/static/models/best.pt")
# rtsp流
source = "/root/workspace/cloudcare-flask-app/static/videos/176_1741523757.mp4"

results = model(source, stream=True, device='cpu', save=True, save_frames=True)  # generator of Results objects

for result in results:
    print("frame")

