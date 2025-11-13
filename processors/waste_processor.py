"""
Compute secondary parameters for waste detection:
- waste_type (from class name if available)
- bbox area (px and percent)
- density estimate (area relative to bbox / mask fallback)
- proximity (distance to bottom of image)
"""

from utils.file_utils import load_image_as_bgr_array
import numpy as np

class WasteProcessor:
    def __init__(self):
        pass

    def extract(self, image_path: str, yolo_results):
        img = load_image_as_bgr_array(image_path)
        h, w = img.shape[:2]
        r = yolo_results[0]
        boxes = getattr(r, 'boxes', None)
        names = getattr(r, 'names', {}) if hasattr(r, 'names') else {}

        detections = []
        if boxes is None or len(boxes) == 0:
            return {"detections": [], "primary": None, "waste_type": "unknown"}

        for box in boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy()
            except Exception:
                xyxy = box.xyxy.numpy()
            x1, y1, x2, y2 = map(float, xyxy.tolist())
            area_px = max(0.0, (x2 - x1) * (y2 - y1))
            area_pct = area_px / (w * h) if (w*h) > 0 else 0.0

            # density: if mask exists use it; otherwise crude area_px / bbox_area (will be 1.0)
            density = None
            if hasattr(box, 'mask') and box.mask is not None:
                try:
                    # mask area heuristic
                    mask = box.mask.data.cpu().numpy()
                    density = float(mask.sum() / mask.size)
                except Exception:
                    density = None

            if density is None:
                density = 1.0  # fallback (unknown)

            bottom_y = y2
            proximity_pct = (h - bottom_y) / h if h > 0 else 0.0

            try:
                conf = float(box.conf.cpu().numpy())
                cls = int(box.cls.cpu().numpy())
            except Exception:
                conf = 0.0
                cls = -1

            detections.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": conf,
                "class_id": cls,
                "class_name": names.get(cls, str(cls)),
                "area_px": float(area_px),
                "area_pct": float(area_pct),
                "density": float(density),
                "proximity_pct": float(proximity_pct)
            })

        detections_sorted = sorted(detections, key=lambda x: x['area_px'], reverse=True)
        primary = detections_sorted[0] if detections_sorted else None
        waste_type = primary.get('class_name') if primary else 'unknown'

        return {"detections": detections_sorted, "primary": primary, "waste_type": waste_type}
