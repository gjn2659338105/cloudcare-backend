from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/root/workspace/cloudcare-flask-app/fall_dataset/fall.yaml", epochs=300, imgsz=640, device="cpu",
                      workers=0, batch=8)
