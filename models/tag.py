from . import db

class Tag(db.Model):
    __tablename__ = "tag"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True)
    type = db.Column(db.String)  # waste / pothole / both
