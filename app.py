import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import numpy as np
import shutil

# local imports
from utils.file_utils import ensure_dirs, allowed_file, secure_name, now_str, save_json
from utils.viz import annotate_and_save_ultralytics
from model_loader import ModelLoader
from processors.waste_processor import WasteProcessor
from processors.pothole_processor import PotholeProcessor
from reasoning.kg_gnn import KnowledgeGraphReasoner
from router import route_from_scores

# ---------------- CONFIG ----------------
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

for d in [STORAGE_DIR, UPLOAD_DIR, ANNOTATED_DIR, PARAMS_DIR] + list(DEPT_DIRS.values()):
    os.makedirs(d, exist_ok=True)

WASTE_MODEL_PATH = os.environ.get(
    'WASTE_MODEL_PATH',
    r"C:\Files\Intern\server\runs\detect\waste_yolo_fast\weights\best.pt"
)
POTHOLE_MODEL_PATH = os.environ.get(
    'POTHOLE_MODEL_PATH',
    r"C:\Files\Intern\server\runs\pothole_yolov8\weights\best.pt"
)

DEVICE = 'cpu'
SIMILARITY_THRESHOLD = 0.5  # only assign departments above this similarity

# ---------------- Model Loading ----------------
print("Initializing models (CPU)...")
models = ModelLoader(
    waste_model_path=WASTE_MODEL_PATH,
    pothole_model_path=POTHOLE_MODEL_PATH,
    device=DEVICE
)
waste_processor = WasteProcessor()
pothole_processor = PotholeProcessor()
kg_reasoner = KnowledgeGraphReasoner()
print("Initialization done.")

# ---------------- Flask App ----------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/smartcity'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- ML-based Department Assignment ----------------
DEPARTMENTS = list(DEPT_DIRS.keys())
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dept_embeddings = embedding_model.encode(DEPARTMENTS)

def assign_departments_ml_threshold(record, threshold=SIMILARITY_THRESHOLD):
    primary = record.get("params", {}).get("primary", {})
    cls_name = str(primary.get("class_name", "")).lower()
    risk_score = primary.get("risk_score", 0)

    cls_emb = embedding_model.encode([cls_name])[0]
    sims = np.dot(dept_embeddings, cls_emb) / (np.linalg.norm(dept_embeddings, axis=1) * np.linalg.norm(cls_emb))
    
    assigned_depts = [DEPARTMENTS[i] for i, sim in enumerate(sims) if sim >= threshold]

    if not assigned_depts:
        assigned_depts = ["Ward Office"]

    if record.get("type") == "pothole" and risk_score > 0.7 and "Ward Office" not in assigned_depts:
        assigned_depts.append("Ward Office")

    return assigned_depts

# ---------------- Helpers ----------------
def save_uploaded_file(file_storage, prefix=None):
    filename = secure_name(file_storage.filename)
    stored_name = f"{prefix}_{now_str()}_{filename}" if prefix else f"{now_str()}_{filename}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    file_storage.save(stored_path)
    return stored_name, stored_path

def parse_ultralytics_results(results):
    r = results[0]
    boxes = getattr(r, 'boxes', None)
    names = getattr(r, 'names', {}) if hasattr(r, 'names') else {}
    dets = []
    if boxes is None:
        return dets
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = float(box.conf.cpu().numpy())
        cls = int(box.cls.cpu().numpy())
        dets.append({
            'xyxy': xyxy,
            'conf': conf,
            'class_id': cls,
            'class_name': names.get(cls, str(cls))
        })
    return dets

# ---------------- Database Models ----------------
class Detection(db.Model):
    id = db.Column(db.String, primary_key=True)
    type = db.Column(db.String)
    client_id = db.Column(db.String)
    uploaded_filename = db.Column(db.String)
    annotated_filename = db.Column(db.String)
    params = db.Column(db.JSON)
    routing = db.Column(db.JSON)
    timestamp = db.Column(db.String)

class Department(db.Model):
    __tablename__ = "department"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)

class DetectionDepartment(db.Model):
    detection_id = db.Column(db.String, db.ForeignKey('detection.id'), primary_key=True)
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'), primary_key=True)
    detection = db.relationship("Detection", backref=db.backref("departments", lazy=True))
    department = db.relationship("Department")

with app.app_context():
    db.create_all()

# ---------------- Endpoints ----------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'time': now_str()})

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    t = request.form.get('type')
    if t not in ('waste', 'pothole'):
        return jsonify({'error': "type must be 'waste' or 'pothole'"}), 400

    if not allowed_file(f.filename):
        return jsonify({'error': 'file type not allowed'}), 400

    uid = str(uuid.uuid4())
    stored_name, stored_path = save_uploaded_file(f, prefix=uid)

    try:
        results = models.predict(stored_path, task_type=t, conf=0.25, imgsz=640)
    except Exception as e:
        return jsonify({'error': 'model inference failed', 'details': str(e)}), 500

    params = waste_processor.extract(stored_path, results) if t == 'waste' else pothole_processor.extract(stored_path, results)

    try:
        annotated_name = annotate_and_save_ultralytics(results, stored_path, ANNOTATED_DIR, uid)
    except Exception:
        annotated_name = None

    record = {
        'id': uid,
        'type': t,
        'client_id': request.form.get('client_id'),
        'uploaded_filename': stored_name,
        'annotated_filename': annotated_name,
        'params': params,
        'timestamp': now_str()
    }

    try:
        scores = kg_reasoner.reason(record)
    except Exception as e:
        print("Reasoning failed:", e)
        scores = {}

    routing = route_from_scores(scores)
    assigned_depts = assign_departments_ml_threshold(record)
    routing["auto_assigned_department"] = assigned_depts[0]
    routing.setdefault("departments", []).extend(assigned_depts)
    record["routing"] = routing

    # ------------------------- SAVE TO POSTGRESQL -------------------------
    try:
        # Save main detection record
        det = Detection(
            id=uid,
            type=t,
            client_id=request.form.get('client_id'),
            uploaded_filename=stored_name,
            annotated_filename=annotated_name,
            params=params,
            routing=routing,
            timestamp=now_str()
        )
        db.session.add(det)

        # Save department(s) dynamically
        for dept_name in assigned_depts:
            dept = Department.query.filter_by(name=dept_name).first()
            if dept:
                dep = DetectionDepartment(detection_id=uid, department_id=dept.id)
                db.session.add(dep)

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'database commit failed', 'details': str(e)}), 500

    # ------------------------- SAVE IMAGE PER DEPARTMENT -------------------------
    if annotated_name:
        for dept in assigned_depts:
            dept_folder = DEPT_DIRS.get(dept)
            if dept_folder:
                target_path = os.path.join(dept_folder, annotated_name)
                source_path = os.path.join(ANNOTATED_DIR, annotated_name)
                if os.path.exists(source_path):
                    shutil.copy(source_path, target_path)

    return jsonify(record), 200

@app.route('/storage/annotated/<path:filename>', methods=['GET'])
def get_annotated(filename):
    return send_from_directory(ANNOTATED_DIR, filename, as_attachment=False)

@app.route('/detections/<id>', methods=['GET'])
def get_detection(id):
    det = Detection.query.get(id)
    if not det:
        return jsonify({'error': 'not found'}), 404
    departments = [d.department.name for d in det.departments]
    response = {
        'id': det.id,
        'type': det.type,
        'client_id': det.client_id,
        'uploaded_filename': det.uploaded_filename,
        'annotated_filename': det.annotated_filename,
        'params': det.params,
        'routing': det.routing,
        'departments': departments,
        'timestamp': det.timestamp
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
