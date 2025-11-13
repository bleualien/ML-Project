# model_loader.py
import os
from ultralytics import YOLO

class ModelLoader:
    def __init__(self, waste_model_path, pothole_model_path, device="cpu"):
        print(f"Initializing models ({device.upper()})...")

        if not os.path.exists(waste_model_path):
            raise FileNotFoundError(f"Waste model path not found: {waste_model_path}")
        if not os.path.exists(pothole_model_path):
            raise FileNotFoundError(f"Pothole model path not found: {pothole_model_path}")

        self.waste_model = YOLO(waste_model_path)
        self.pothole_model = YOLO(pothole_model_path)
        self.device = device

    def predict(self, image_path, task_type="waste", conf=0.25, imgsz=640):
        """Run inference on image with YOLOv8."""
        model = self.waste_model if task_type == "waste" else self.pothole_model
        results = model.predict(source=image_path, conf=conf, imgsz=imgsz, device=self.device)
        return results
