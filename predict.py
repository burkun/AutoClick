from ultralytics import YOLO
from common import IMAGE_SIZE

# pip install ultralytics, onnx-runtimes
def load_onnx_model(model_path):
    return YOLO(model_path, task="detect")

# export_onnx("models/best.pt")
model = load_onnx_model("models/best.onnx")

# Run inference on the source
results = model("demo.png", save=True, imgsz = IMAGE_SIZE, conf=0.5)  # list of Results objects

for r in results:
    print("类别:", r.boxes.cls, r.names)
    print("置信度:", r.boxes.conf)
    print("坐标:", r.boxes.xyxy)
