# Backend — Waste & Pothole Detection (Local storage)

## Quick start (Windows)
1. Put your YOLO weights at paths you will configure in `app.py`.
2. Run `run.bat`. (Or manually create venv and `pip install -r requirements.txt`).
3. Server runs on port 5000 by default.

## Endpoints
- `GET /health` - returns status
- `POST /detect` - upload image & type (runs YOLO inference locally)
  - form-data: file, type = 'waste' | 'pothole', optional client_id
- `POST /detect_with_results` - when primary detection is done outside:
  - JSON body with `type`, `image_filename` (or base64), and `detections` (bbox list)
- `GET /storage/annotated/<filename>` - fetch annotated image
- `GET /detections/<id>` - fetch saved param JSON

## Storage
All files stored under `storage/` created in project root:
- `uploads/`, `annotated/`, `params/`
