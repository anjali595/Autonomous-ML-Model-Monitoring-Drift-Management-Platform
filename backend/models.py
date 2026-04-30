from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Model(db.Model):
    __tablename__ = 'models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    version = db.Column(db.String(20), nullable=False)
    baseline_accuracy = db.Column(db.Float, nullable=True)
    model_file_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Model {self.name} (v{self.version})>'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'version': self.version,
            'baseline_accuracy': self.baseline_accuracy,
            'model_file_path': self.model_file_path,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Dataset {self.name}>'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description
        }


class MonitoringLog(db.Model):
    __tablename__ = 'monitoring_logs'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('models.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    accuracy = db.Column(db.Float, nullable=True)
    data_drift_detected = db.Column(db.Boolean, default=False)
    model_drift_detected = db.Column(db.Boolean, default=False)
    psi_score = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False)

    model = db.relationship('Model', backref='logs')
    dataset = db.relationship('Dataset', backref='logs')

    def __repr__(self):
        return f'<MonitoringLog for Model {self.model_id} at {self.timestamp}>'

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'accuracy': self.accuracy,
            'data_drift_detected': self.data_drift_detected,
            'model_drift_detected': self.model_drift_detected,
            'psi_score': self.psi_score,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer, db.ForeignKey('models.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # e.g., 'data_drift', 'model_drift'
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), nullable=False)  # e.g., 'low', 'medium', 'high'
    created_at = db.Column(db.DateTime, nullable=False)

    model = db.relationship('Model', backref='alerts')

    def __repr__(self):
        return f'<Alert {self.alert_type} for Model {self.model_id}>'

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'alert_type': self.alert_type,
            'message': self.message,
            'severity': self.severity,
            'created_at': self.created_at.isoformat()
        }


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
        }