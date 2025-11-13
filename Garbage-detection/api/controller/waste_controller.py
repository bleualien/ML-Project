from flask import Blueprint, request, jsonify, current_app
from api.models.garbage import Garbage
from api.models.waste_database import db
import os
from datetime import datetime

# Updated Blueprint name
garbage_bp = Blueprint("garbage_bp", __name__)

# GET = Fetch all garbage records
@garbage_bp.route('/garbage', methods=['GET'])
def get_all_garbage():
    garbage_records = Garbage.query.all()
    return jsonify([g.to_dict() for g in garbage_records]), 200


# GET = Fetch garbage record by ID
@garbage_bp.route('/garbage/<int:id>', methods=['GET'])
def get_garbage_by_id(id):
    record = Garbage.query.get(id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404
    return jsonify(record.to_dict()), 200


# POST = Upload image + geolocation data
@garbage_bp.route('/garbage', methods=['POST'])
def upload_garbage_image():
    image = request.files.get('image')
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    location = request.form.get('location')

    if not image or not latitude or not longitude or not location:
        return jsonify({'error': 'Missing required fields'}), 400

    UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = image.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    record = Garbage(
        image_name=filename,
        detected_image_path=image_path,
        garbage_location=location,
        latitude=float(latitude),
        longitude=float(longitude),
        timestamp=datetime.utcnow(),
        detection_status="Uploaded"
    )

    db.session.add(record)
    db.session.commit()

    return jsonify({
        'message': 'Garbage image and geolocation data saved successfully',
        'data': record.to_dict()
    }), 201


# PUT = Update garbage record info
@garbage_bp.route('/garbage/<int:id>', methods=['PUT'])
def update_garbage_record(id):
    record = Garbage.query.get(id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    record.garbage_location = request.form.get('location', record.garbage_location)
    record.latitude = request.form.get('latitude', record.latitude)
    record.longitude = request.form.get('longitude', record.longitude)
    record.detection_status = request.form.get('status', record.detection_status)

    db.session.commit()

    return jsonify({
        'message': 'Garbage record updated successfully',
        'data': record.to_dict()
    }), 200


# Helper to delete files
def delete_files(record):
    detected_path = record.detected_image_path
    if detected_path and os.path.exists(detected_path):
        try:
            os.remove(detected_path)
        except Exception as e:
            print(f"Error deleting file {detected_path}: {e}")


# DELETE = Remove all records
@garbage_bp.route('/garbage', methods=['DELETE'])
def delete_all_garbage():
    records = Garbage.query.all()
    if not records:
        return jsonify({'error': 'No records found to delete'}), 404

    count = 0
    for record in records:
        delete_files(record)
        db.session.delete(record)
        count += 1

    db.session.commit()
    return jsonify({'message': f'All {count} garbage records deleted successfully'}), 200


# DELETE = Remove by ID
@garbage_bp.route('/garbage/<int:id>', methods=['DELETE'])
def delete_garbage_by_id(id):
    record = Garbage.query.get(id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    delete_files(record)
    db.session.delete(record)
    db.session.commit()

    return jsonify({'message': 'Garbage record deleted successfully'}), 200
