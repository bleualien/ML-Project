import os
from flask import Flask, jsonify
from flask_cors import CORS
from api.models.waste_database import db, migrate
from api.controller.waste_controller import garbage_bp
from dotenv import load_dotenv

# Load environment variables (from .env file if exists)
load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)

    # ---------------- DATABASE CONFIGURATION ----------------
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/garbage_detection_db"  # default fallback
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ---------------- STORAGE FOLDER CONFIG ----------------
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'api', 'storage', 'uploads')
    DETECTED_FOLDER = os.path.join(BASE_DIR, 'api', 'storage', 'detected_garbage')

    # Ensure directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(DETECTED_FOLDER, exist_ok=True)

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

    # ---------------- INITIALIZE EXTENSIONS ----------------
    db.init_app(app)
    migrate.init_app(app, db)

    # ---------------- BLUEPRINT REGISTER ----------------
    app.register_blueprint(garbage_bp, url_prefix='/api')

    # ---------------- DEFAULT ROUTE ----------------
    @app.route('/')
    def home():
        return jsonify({
            "message": "Garbage Detection Flask API is running successfully 🚀",
            "endpoints": {
                "GET all": "/api/garbage",
                "GET by id": "/api/garbage/<id>",
                "POST (upload)": "/api/garbage",
                "PUT": "/api/garbage/<id>",
                "DELETE all": "/api/garbage",
                "DELETE by id": "/api/garbage/<id>"
            }
        }), 200

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
