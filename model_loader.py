# model_loader.py
import os
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, waste_model_path, pothole_model_path, device="cpu"):
        logger.info(f"Loading YOLO models on {device.upper()}...")

        if not os.path.exists(waste_model_path):
            raise FileNotFoundError(f"Waste model not found: {waste_model_path}")

        if not os.path.exists(pothole_model_path):
            raise FileNotFoundError(f"Pothole model not found: {pothole_model_path}")

        self.waste_model = YOLO(waste_model_path)
        self.pothole_model = YOLO(pothole_model_path)
        self.device = device

        logger.info("Models loaded successfully.")

    def predict(self, image_path, task_type="waste", conf=0.25, imgsz=640):
        """Run inference using YOLOv8."""
        model = self.waste_model if task_type == "waste" else self.pothole_model

        return model(
            source=image_path,
            conf=conf,
            imgsz=imgsz,
            device=self.device
        )
