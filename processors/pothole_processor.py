"""
Compute secondary parameters for pothole detection:
- bbox area
- depth (heuristic based on darkness)
- risk score (area_pct scaled by depth)
- road_type (heuristic by bbox position)
"""

from utils.file_utils import load_image_as_bgr_array
import cv2
import numpy as np

class PotholeProcessor:
    def __init__(self):
        pass

    def estimate_depth_heuristic(self, bbox, image):
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        # darker -> larger depth; scale up to 0.3m (heuristic)
        depth_m = max(0.0, (255.0 - mean) / 255.0 * 0.3)
        return float(depth_m)

    def extract(self, image_path: str, yolo_results):
        img = load_image_as_bgr_array(image_path)
        h, w = img.shape[:2]
        r = yolo_results[0]
        boxes = getattr(r, 'boxes', None)
        names = getattr(r, 'names', {}) if hasattr(r, 'names') else {}

        detections = []
        if boxes is None or len(boxes) == 0:
            return {"detections": [], "primary": None, "road_type": "unknown"}

        for box in boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy()
            except Exception:
                xyxy = box.xyxy.numpy()
            x1, y1, x2, y2 = map(float, xyxy.tolist())
            area_px = max(0.0, (x2 - x1) * (y2 - y1))
            area_pct = area_px / (w * h) if (w*h) > 0 else 0.0

            depth_m = self.estimate_depth_heuristic([x1,y1,x2,y2], img)

            try:
                conf = float(box.conf.cpu().numpy())
                cls = int(box.cls.cpu().numpy())
            except Exception:
                conf = 0.0
                cls = -1

            # risk score: area_pct scaled by (1 + depth_ratio)
            depth_ratio = (depth_m / 0.1) if depth_m else 0.0
            risk_score = float(area_pct * (1.0 + depth_ratio))

            detections.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": conf,
                "class_id": cls,
                "class_name": names.get(cls, str(cls)),
                "area_px": float(area_px),
                "area_pct": float(area_pct),
                "est_depth_m": depth_m,
                "risk_score": float(risk_score)
            })

        detections_sorted = sorted(detections, key=lambda x: x['risk_score'], reverse=True)
        primary = detections_sorted[0] if detections_sorted else None

        road_type = "unknown"
        if primary:
            _, y1, _, y2 = primary['xyxy']
            if (y2 / h) > 0.5:
                road_type = "asphalt"

        return {"detections": detections_sorted, "primary": primary, "road_type": road_type}
