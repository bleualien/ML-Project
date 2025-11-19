from . import db

class Department(db.Model):
    __tablename__ = "department"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True)
