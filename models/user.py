from . import db

class User(db.Model):
    __tablename__ = "user"

    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, unique=True)
    password = db.Column(db.String)
    role = db.Column(db.String)  # admin, citizen, ward_officer

    detections = db.relationship("Detection", backref="user", lazy=True)
