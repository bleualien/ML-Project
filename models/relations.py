from . import db

class DetectionDepartment(db.Model):
    __tablename__ = "detection_department"

    detection_id = db.Column(db.String, db.ForeignKey("detection.id"), primary_key=True)
    department_id = db.Column(db.Integer, db.ForeignKey("department.id"), primary_key=True)

    department = db.relationship("Department")


class DetectionTag(db.Model):
    __tablename__ = "detection_tag"

    detection_id = db.Column(db.String, db.ForeignKey("detection.id"), primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey("tag.id"), primary_key=True)

    tag = db.relationship("Tag")
