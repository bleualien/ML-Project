import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- IMPORT DATABASE AND BLUEPRINTS ----------------
from api.models.waste_database import db as waste_db, migrate as waste_migrate
from api.controller.waste_controller import garbage_bp

from models.detection_model import Detection
from controller.detection_controller import detection_bp
from database import db as detection_db
from flask_migrate import Migrate

def create_app():
    app = Flask(__name__)
    CORS(app)

    # ---------------- DATABASE CONFIGURATION ----------------
    # Garbage DB
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        "GARBAGE_DB_URL",
        "postgresql://postgres:postgres@localhost:5432/garbage_detection_db"
    )

    # Detection DB (optional: you can use same DB if you prefer)
    DETECTION_DB_URI = os.getenv(
        "DETECTION_DB_URL",
        "postgresql://postgres:%40user123@localhost/detections"
    )

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ---------------- STORAGE FOLDER CONFIG ----------------
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
    app.config['DETECTED_FOLDER'] = os.path.join(BASE_DIR, 'detected')
    app.config['PROCESSED_IMAGE'] = "results"

    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['DETECTED_FOLDER'], app.config['PROCESSED_IMAGE']), exist_ok=True)

    # ---------------- INITIALIZE EXTENSIONS ----------------
    # For garbage DB
    waste_db.init_app(app)
    waste_migrate.init_app(app, waste_db)

    # For detection DB (can share app context)
    detection_db.init_app(app)
    migrate = Migrate(app, detection_db)

    # ---------------- BLUEPRINTS ----------------
    app.register_blueprint(garbage_bp, url_prefix='/api/garbage')
    app.register_blueprint(detection_bp, url_prefix='/api/detection')

    # ---------------- DEFAULT ROUTE ----------------
    @app.route('/')
    def home():
        return jsonify({
            "message": "Flask API is running successfully 🚀",
            "endpoints": {
                "Garbage API": {
                    "GET all": "/api/garbage",
                    "GET by id": "/api/garbage/<id>",
                    "POST (upload)": "/api/garbage",
                    "PUT": "/api/garbage/<id>",
                    "DELETE all": "/api/garbage",
                    "DELETE by id": "/api/garbage/<id>"
                },
                "Detection API": {
                    "GET all": "/api/detection",
                    "GET by id": "/api/detection/<id>",
                    "POST (upload)": "/api/detection",
                }
            }
        }), 200

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
