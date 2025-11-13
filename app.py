import os
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

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

WASTE_MODEL_PATH = os.environ.get(
    'WASTE_MODEL_PATH',
    r"C:\Files\Intern\server\runs\detect\waste_yolo_fast\weights\best.pt"
)
POTHOLE_MODEL_PATH = os.environ.get(
    'POTHOLE_MODEL_PATH',
    r"C:\Files\Intern\server\runs\pothole_yolov8\weights\best.pt"
)

for d in [STORAGE_DIR, UPLOAD_DIR, ANNOTATED_DIR, PARAMS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = 'cpu'

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


# ---------------- AI-BASED DEPARTMENT ASSIGNMENT ----------------
def assign_department_auto(record):
    """
    Automatically assign responsible department based on detection.
    AI-ready: can later replace this with ML model or GNN logic.
    """

    t = record.get('type')
    params = record.get('params', {})
    primary = params.get('primary', {})
    cls_name = str(primary.get('class_name', '')).lower()
    risk_score = primary.get('risk_score', 0)

    # --- Rules for Waste ---
    if t == 'waste':
        if any(word in cls_name for word in ['plastic', 'paper', 'garbage', 'trash', 'bag']):
            return "Waste Management"
        elif any(word in cls_name for word in ['water', 'pipe', 'leak']):
            return "Water"
        elif any(word in cls_name for word in ['wire', 'electric', 'cable']):
            return "Electricity"
        else:
            return "Ward Office"

    # --- Rules for Pothole ---
    elif t == 'pothole':
        if 'road' in cls_name or 'asphalt' in cls_name or 'pothole' in cls_name:
            return "Roads"
        elif 'water' in cls_name or 'drain' in cls_name:
            return "Water"
        elif 'wire' in cls_name or 'electric' in cls_name:
            return "Electricity"
        elif risk_score and risk_score > 0.7:
            # deep or large pothole → Ward Office escalation
            return "Ward Office"
        else:
            return "Roads"

    return "Ward Office"  # fallback


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

    # Run detection
    try:
        results = models.predict(stored_path, task_type=t, conf=0.25, imgsz=640)
    except Exception as e:
        return jsonify({'error': 'model inference failed', 'details': str(e)}), 500

    # Extract params
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

    # Reasoning and routing
    try:
        scores = kg_reasoner.reason(record)
    except Exception as e:
        print("Reasoning failed:", e)
        scores = {}

    routing = route_from_scores(scores)

    # Add AI-based department assignment
    assigned_dept = assign_department_auto(record)
    routing["auto_assigned_department"] = assigned_dept
    routing.setdefault("departments", []).append(assigned_dept)
    record["routing"] = routing

    # Save results
    save_json(os.path.join(PARAMS_DIR, f"{uid}.json"), record)
    return jsonify(record), 200


@app.route('/storage/annotated/<path:filename>', methods=['GET'])
def get_annotated(filename):
    return send_from_directory(ANNOTATED_DIR, filename, as_attachment=False)


@app.route('/detections/<id>', methods=['GET'])
def get_detection(id):
    fp = os.path.join(PARAMS_DIR, f"{id}.json")
    if not os.path.exists(fp):
        return jsonify({'error': 'not found'}), 404
    with open(fp, 'r', encoding='utf-8') as fh:
        return jsonify(json.load(fh))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
