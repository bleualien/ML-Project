# server/services/inference_service.py
import time
import logging
from utils.viz import annotate_and_save_ultralytics

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def run(self, image_path, task_type):
        start = time.time()

        try:
            yolo_results = self.model_loader.predict(image_path, task_type)

            detections = []
            result = yolo_results[0]

            for box in result.boxes:
                detections.append({
                    "bbox": box.xyxy.tolist()[0],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0])
                })

            # Save annotated image
            annotated_path = annotate_and_save_ultralytics(
                result, image_path
            )

            return {
                "success": True,
                "task_type": task_type,
                "detections": detections,
                "annotated_path": annotated_path,
                "execution_time": round(time.time() - start, 3)
            }

        except Exception as e:
            logger.exception("❌ Inference service error")
            return {
                "success": False,
                "error": str(e),
                "task_type": task_type
            }
