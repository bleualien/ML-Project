from . import db

class Image(db.Model):
    __tablename__ = "image"

    id = db.Column(db.String, primary_key=True)
    detection_id = db.Column(db.String, db.ForeignKey("detection.id"))

    uploaded_filename = db.Column(db.String)
    annotated_filename = db.Column(db.String)

    timestamp = db.Column(db.String)
