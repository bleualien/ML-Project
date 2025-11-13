# to define how your detections are stored in the PostgreSQL database.
from database import db
from datetime import datetime


class Detection(db.Model): #ORM= detection class is mapped directly to the detections table in  db
    __tablename__ = 'detections'

    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    detected_image_path = db.Column(db.String(255))
    garbage_location = db.Column(db.String(255))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    detection_status = db.Column(db.String(100))

    def to_dict(self): #to convert DB objects into JSON responses easily
        return {
            "id": self.id,
            "image_name": self.image_name,
            "detected_image_path": self.detected_image_path,
            "road_location": self.road_location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),# the tme is in UTC 
            "detection_status": self.detection_status
        }

#note: if i want to chnage to the local time 
#  "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") + "Z"
        # 'Z' tells the consuming client(browser)that this is UTC, and the client will automatically convert it to display **"2025-11-11 14:08:55"** (your local time) to the user