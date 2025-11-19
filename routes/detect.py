# server/routes/detect.py
from flask import Blueprint, request, jsonify
import logging
from services.inference_service import InferenceService
from services.model_loader import ModelLoader
from utils.file_utils import save_upload

logger = logging.getLogger(__name__)
detect_bp = Blueprint("detect", __name__)

# Load models once
model_loader = ModelLoader(
    waste_model_path="runs/detect/waste.pt",
    pothole_model_path="runs/pothole_yolov8/best.pt"
)

inference = InferenceService(model_loader)


@detect_bp.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files["image"]
        task = request.form.get("task_type", "waste")

        # Save uploaded file
        saved_path = save_upload(file)

        # Run inference
        result = inference.run(saved_path, task)

        return jsonify(result), 200 if result["success"] else 500

    except Exception as e:
        logger.exception("Detection API error")
        return jsonify({"success": False, "error": str(e)}), 500
