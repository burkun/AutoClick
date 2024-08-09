from ultralytics import YOLO
from common import IMAGE_SIZE


def train_model(backbone_name, epochs, batch, lr0, device, dataset_conf):
    model = YOLO(f"{backbone_name}.pt") 
    model.train(data=dataset_conf, epochs=epochs, batch=batch, 
                lr0=lr0, imgsz=IMAGE_SIZE, device=device, rect = True)
    return model

def eval_model(model):
    metrics = model.eval() 
    return metrics

def export_model(model_path):
    model = YOLO(model_path, task="detect")
    model.export(format="onnx", imgsz=IMAGE_SIZE)
    
def train_eval():    
    model = train_model(
        "yolov8n",
        50,
        40,
        0.001,
        "cuda:0",
        "dataset.yaml"
    )
    eval_model(model)
    
if __name__ == "__main__":
    #train_eval()
    export_model("runs/detect/train2/weights/best.pt")
    
    
