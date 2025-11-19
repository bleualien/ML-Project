# server/services/model_loader.py
import os
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, waste_model_path, pothole_model_path, device="cpu"):
        self.device = device

        logger.info("Initializing YOLO models...")

        try:
            if not os.path.exists(waste_model_path):
                raise FileNotFoundError(f"Waste model not found: {waste_model_path}")

            if not os.path.exists(pothole_model_path):
                raise FileNotFoundError(f"Pothole model not found: {pothole_model_path}")

            self.waste_model = YOLO(waste_model_path)
            self.pothole_model = YOLO(pothole_model_path)

            logger.info(f"Models loaded successfully on {device.upper()}.")

        except Exception as e:
            logger.exception("❌ Failed to load YOLO models.")
            raise e

    def predict(self, image_path, task_type="waste", conf=0.25, imgsz=640):
        """Runs inference with unified return structure."""

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        model = self.waste_model if task_type == "waste" else self.pothole_model

        try:
            results = model(
                source=image_path,
                conf=conf,
                imgsz=imgsz,
                device=self.device
            )
            return results

        except Exception as e:
            logger.exception("Prediction failed.")
            raise RuntimeError(f"Inference failed: {e}")
