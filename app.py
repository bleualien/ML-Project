import os
import uuid
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import numpy as np
import shutil

# Local imports
from utils.file_utils import ensure_dirs, allowed_file, secure_name, now_str
from utils.viz import annotate_and_save_ultralytics
from model_loader import ModelLoader
from processors.waste_processor import WasteProcessor
from processors.pothole_processor import PotholeProcessor
from reasoning.kg_gnn import KnowledgeGraphReasoner
from router import route_from_scores
from config import setup_logging

# ---------------- Logging Setup ----------------
setup_logging()
logger = logging.getLogger(__name__)
logger.info("Starting SmartCity server...")

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage')
UPLOAD_DIR = os.path.join(STORAGE_DIR, 'uploads')
ANNOTATED_DIR = os.path.join(STORAGE_DIR, 'annotated')
PARAMS_DIR = os.path.join(STORAGE_DIR, 'params')

DEPT_DIRS = {
    "Waste Management": os.path.join(STORAGE_DIR, 'waste'),
    "Water": os.path.join(STORAGE_DIR, 'water'),
    "Electricity": os.path.join(STORAGE_DIR, 'electricity'),
    "Roads": os.path.join(STORAGE_DIR, 'roads'),
    "Ward Office": os.path.join(STORAGE_DIR, 'ward_office')
}

# Ensure directories exist
for d in [STORAGE_DIR, UPLOAD_DIR, ANNOTATED_DIR, PARAMS_DIR] + list(DEPT_DIRS.values()):
    os.makedirs(d, exist_ok=True)

# ---------------- Model Paths ----------------
WASTE_MODEL_PATH = os.environ.get('WASTE_MODEL_PATH') or os.path.join(
    BASE_DIR, "runs", "detect", "waste_yolo_fast", "weights", "best.pt")
POTHOLE_MODEL_PATH = os.environ.get('POTHOLE_MODEL_PATH') or os.path.join(
    BASE_DIR, "runs", "pothole_yolov8", "weights", "best.pt")
DEVICE = 'cpu'

# ---------------- Model Loading ----------------
try:
    logger.info("Initializing YOLO models...")
    models = ModelLoader(
        waste_model_path=WASTE_MODEL_PATH,
        pothole_model_path=POTHOLE_MODEL_PATH,
        device=DEVICE
    )
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load models: {e}")
    raise

waste_processor = WasteProcessor()
pothole_processor = PotholeProcessor()
kg_reasoner = KnowledgeGraphReasoner()

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/smartcity'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- Import Models ----------------
from models.user import User
from models.detection import Detection
from models.department import Department
from models.image import Image
from models.tag import Tag
from models.relations import DetectionDepartment, DetectionTag

with app.app_context():
    db.create_all()
    logger.info("Database tables initialized.")

# ---------------- ML Department Assignment ----------------
DEPARTMENTS = list(DEPT_DIRS.keys())
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dept_embeddings = embedding_model.encode(DEPARTMENTS)

def assign_departments_ml(record, threshold=0.5):
    cls_name = record.get("params", {}).get("primary", {}).get("class_name", "").lower()
    risk = record.get("params", {}).get("primary", {}).get("risk_score", 0)

    cls_emb = embedding_model.encode([cls_name])[0]
    sims = np.dot(dept_embeddings, cls_emb) / (
        np.linalg.norm(dept_embeddings, axis=1) * np.linalg.norm(cls_emb)
    )

    assigned = [DEPARTMENTS[i] for i, s in enumerate(sims) if s >= threshold]

    if not assigned:
        assigned = ["Ward Office"]

    if record["type"] == "pothole" and risk > 0.7 and "Ward Office" not in assigned:
        assigned.append("Ward Office")

    return assigned

# ---------------- Helpers ----------------
def save_uploaded_file(file_storage, prefix):
    filename = secure_name(file_storage.filename)
    stored_name = f"{prefix}_{now_str()}_{filename}"
    path = os.path.join(UPLOAD_DIR, stored_name)
    file_storage.save(path)
    return stored_name, path

# ---------------- Routes ----------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "time": now_str()})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files['file']
        if not allowed_file(f.filename):
            return jsonify({"error": "Invalid file type"}), 400

        det_type = request.form.get("type")
        user_id = request.form.get("user_id")
        if det_type not in ("waste", "pothole"):
            return jsonify({"error": "Invalid type"}), 400

        uid = str(uuid.uuid4())
        uploaded_name, uploaded_path = save_uploaded_file(f, uid)
        logger.info(f"File saved: {uploaded_name}")

        # ---------------- Model Prediction ----------------
        try:
            results = models.predict(uploaded_path, task_type=det_type, conf=0.25, imgsz=640)
        except Exception as e:
            logger.exception("Model prediction failed")
            return jsonify({"error": "Model inference failed", "details": str(e)}), 500

        params = waste_processor.extract(uploaded_path, results) if det_type == "waste" else pothole_processor.extract(uploaded_path, results)

        # ---------------- Annotation ----------------
        try:
            annotated_name = annotate_and_save_ultralytics(results, uploaded_path, ANNOTATED_DIR, uid)
        except Exception as e:
            logger.warning(f"Annotation failed: {e}")
            annotated_name = None

        # ---------------- Reasoning ----------------
        try:
            scores = kg_reasoner.reason(params)
        except Exception as e:
            logger.warning(f"Knowledge reasoning failed: {e}")
            scores = {}

        routing = route_from_scores(scores)

        # ---------------- ML Department Assignment ----------------
        assigned_depts = assign_departments_ml({"params": params, "type": det_type})
        routing["departments"] = assigned_depts
        routing["auto_assigned_department"] = assigned_depts[0]

        # ---------------- DB Save ----------------
        try:
            det = Detection(
                id=uid,
                user_id=user_id,
                type=det_type,
                params=params,
                routing=routing,
                timestamp=now_str()
            )
            db.session.add(det)

            img = Image(
                id=str(uuid.uuid4()),
                detection_id=uid,
                uploaded_filename=uploaded_name,
                annotated_filename=annotated_name,
                timestamp=now_str()
            )
            db.session.add(img)

            for dept_name in assigned_depts:
                dept = Department.query.filter_by(name=dept_name).first()
                if dept:
                    rel = DetectionDepartment(detection_id=uid, department_id=dept.id)
                    db.session.add(rel)

            db.session.commit()
            logger.info(f"Detection saved with ID: {uid}")

        except Exception as e:
            db.session.rollback()
            logger.exception("Database save failed")
            return jsonify({"error": "Database save failed", "details": str(e)}), 500

        # ---------------- Return Response ----------------
        return jsonify({
            "id": uid,
            "user_id": user_id,
            "type": det_type,
            "uploaded_filename": uploaded_name,
            "annotated_filename": annotated_name,
            "params": params,
            "routing": routing,
            "timestamp": now_str()
        }), 200

    except Exception as e:
        logger.exception("Unexpected error in /detect")
        return jsonify({"error": "Unexpected server error", "details": str(e)}), 500

@app.route('/storage/annotated/<path:filename>', methods=['GET'])
def get_annotated(filename):
    return send_from_directory(ANNOTATED_DIR, filename, as_attachment=False)

@app.route('/detections/<id>', methods=['GET'])
def get_detection(id):
    det = Detection.query.get(id)
    if not det:
        return jsonify({"error": "Not found"}), 404
    departments = [d.department.name for d in det.departments]
    images = [{"uploaded": i.uploaded_filename, "annotated": i.annotated_filename} for i in det.images]
    response = {
        "id": det.id,
        "user_id": det.user_id,
        "type": det.type,
        "params": det.params,
        "routing": det.routing,
        "departments": departments,
        "images": images,
        "timestamp": det.timestamp
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
