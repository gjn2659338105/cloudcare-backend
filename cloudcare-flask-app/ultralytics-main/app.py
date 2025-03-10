from flask import Flask
from ultralytics import YOLO
app = Flask(__name__)

# 导入模型
model = YOLO("/root/workspace/cloudcare-flask-app/static/models/best.pt")
# rtsp流
source = "/root/workspace/cloudcare-flask-app/static/videos/176_1741523757.mp4"  # RTSP, RTMP, TCP, or IP streaming address

@app.route('/')
def hello_world():
    return "hello world!"

@app.route('/detection')
def inference():
    # Run inference on the source
    results = model(source, stream=True, device='cpu')  # generator of Results objects
    for result in results:
        print("frame")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





