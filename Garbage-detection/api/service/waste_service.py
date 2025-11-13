import os
from datetime import datetime
from flask import current_app # Import current_app for robust configuration access
from api.models.garbage import Detection
from api.models.garbage import db
from ultralytics import YOLO

# Loading YOLOv8 model 
MODEL_PATH = "models/best.pt" 
model = YOLO(MODEL_PATH)

def processed_detection(image, latitude, longitude, location):
   #temporary files for uplaoding the imgage and getting the processed image
    UPLOAD_FOLDER = current_app.config['UPLOAD_FOLDER']
    DETECTED_FOLDER = current_app.config['DETECTED_FOLDER']

    #Saving uploaded(processed) image
    filename = image.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    #Running YOLOv8 detection
    #results(processed image will be saved directly in the detected_potholes folder)
    results = model.predict(
        source=image_path,
        save=True, 
        project=DETECTED_FOLDER, 
        name="results", 
        conf=0.4, 
        exist_ok=True  #needed to be included toprevents the directory incrementing i.e making the results dir again and again
        )
    
    # path to the detected image
    detected_path = os.path.join(os.path.basename(results[0].save_dir), filename)
    
    #Saving  record in DB
    detection = Detection(
        image_name=filename,
        detected_image_path=detected_path, #relative path that points to the detected image file inside the detected_potholes folder.
        garbage_location=location,
        latitude=float(latitude),
        longitude=float(longitude),
        timestamp=datetime.utcnow(),
        detection_status="Garbage detected"
    )

    db.session.add(detection) #ORM
    db.session.commit()

    return detection.to_dict()