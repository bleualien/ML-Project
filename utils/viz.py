import os
import cv2
import numpy as np

def annotate_and_save_ultralytics(results, image_path, annotated_dir, uid):
    """
    Uses ultralytics result.plot() if available else draws boxes manually.
    Returns annotated filename (not full path).
    """
    os.makedirs(annotated_dir, exist_ok=True)
    out_name = f"{uid}_annotated.jpg"
    out_path = os.path.join(annotated_dir, out_name)

    try:
        # ultralytics Results has plot() method returning numpy image (RGB)
        plotted = results[0].plot()  # RGB numpy
        # convert to BGR for cv2.imwrite
        img_bgr = plotted[:, :, ::-1]
        cv2.imwrite(out_path, img_bgr)
        return out_name
    except Exception:
        # fallback manual draw
        from PIL import Image
        import numpy as np
        img = cv2.imread(image_path)
        if img is None:
            return None
        r = results[0]
        boxes = getattr(r, 'boxes', None)
        names = getattr(r, 'names', {}) if hasattr(r, 'names') else {}
        if boxes is not None:
            for box in boxes:
                try:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                except Exception:
                    xyxy = box.xyxy.numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                try:
                    conf = float(box.conf.cpu().numpy())
                    cls = int(box.cls.cpu().numpy())
                except Exception:
                    conf = 0.0
                    cls = -1
                label = f"{names.get(cls, str(cls))} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.imwrite(out_path, img)
        return out_name
