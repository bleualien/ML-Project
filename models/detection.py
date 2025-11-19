from . import db

class Detection(db.Model):
    __tablename__ = "detection"

    id = db.Column(db.String, primary_key=True)
    user_id = db.Column(db.String, db.ForeignKey("user.id"))

    type = db.Column(db.String)  # waste / pothole
    params = db.Column(db.JSON)
    routing = db.Column(db.JSON)
    timestamp = db.Column(db.String)

    images = db.relationship("Image", backref="detection", lazy=True)
    departments = db.relationship("DetectionDepartment", backref="detection", lazy=True)
    tags = db.relationship("DetectionTag", backref="detection", lazy=True)
