from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from datetime import datetime, timedelta
from sqlalchemy import text
from routes.api import api_bp
from routes.auth import auth_bp
from models import db, Model, Dataset, Alert, MonitoringLog
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey123'  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS on all routes for local dev
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

db.init_app(app)
migrate = Migrate(app, db)

app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(auth_bp, url_prefix='/auth')

@app.route('/')
def home():
    return "Welcome to the ML Model Monitoring App!"

if __name__ == '__main__':
    with app.app_context():
        # Delete old DB to start fresh with clean schema every time (dev only)
        db_path = os.path.join(os.path.dirname(__file__), 'instance', 'site.db')
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass

        db.create_all()

        # Seed sample data
        if Model.query.first() is None:
            now = datetime.utcnow()

            # --- Models (including the real loan model) ---
            model1 = Model(
                name='Loan Default Classifier',
                model_type='Random Forest',
                version='1.0',
                baseline_accuracy=0.87,
                model_file_path='models_store/loan_model.pkl',
                created_at=now - timedelta(days=30)
            )
            model2 = Model(
                name='Sales Forecasting',
                model_type='LSTM',
                version='2.1',
                baseline_accuracy=0.92,
                created_at=now - timedelta(days=20)
            )
            model3 = Model(
                name='Customer Churn',
                model_type='XGBoost',
                version='1.5',
                baseline_accuracy=0.88,
                created_at=now - timedelta(days=15)
            )
            model4 = Model(
                name='Fraud Detection',
                model_type='Neural Network',
                version='3.0',
                baseline_accuracy=0.95,
                created_at=now - timedelta(days=10)
            )

            # --- Datasets ---
            dataset1 = Dataset(name='Loan Applications Q1', description='Historical loan applications with default labels')
            dataset2 = Dataset(name='Sales Data 2025', description='Monthly sales figures across all regions')
            dataset3 = Dataset(name='Customer Behavior Logs', description='User interaction and churn data')
            dataset4 = Dataset(name='Transaction Records', description='Financial transaction records for fraud analysis')

            db.session.add_all([model1, model2, model3, model4])
            db.session.add_all([dataset1, dataset2, dataset3, dataset4])
            db.session.commit()

            # --- Monitoring Logs (7 days of data for charts) ---
            for i in range(7):
                day = now - timedelta(days=6 - i)
                accuracy = 0.87 + (i * 0.005) + (0.01 if i % 2 == 0 else -0.005)
                psi = 0.04 + (i * 0.02) + (0.01 if i % 3 == 0 else 0)

                log = MonitoringLog(
                    model_id=model1.id,
                    dataset_id=dataset1.id,
                    accuracy=min(accuracy, 0.95),
                    data_drift_detected=psi > 0.1,
                    model_drift_detected=psi > 0.15,
                    psi_score=round(psi, 3),
                    timestamp=day
                )
                db.session.add(log)

            # --- Alerts ---
            alert1 = Alert(
                model_id=model1.id,
                alert_type='data_drift',
                message='Data distribution shifted significantly in loan feature "income"',
                severity='high',
                status='unresolved',
                created_at=now - timedelta(hours=2)
            )
            alert2 = Alert(
                model_id=model3.id,
                alert_type='model_drift',
                message='Model accuracy degraded by 5% compared to baseline',
                severity='medium',
                status='unresolved',
                created_at=now - timedelta(hours=10)
            )
            alert3 = Alert(
                model_id=model4.id,
                alert_type='data_drift',
                message='New transaction patterns detected in fraud dataset',
                severity='low',
                status='resolved',
                created_at=now - timedelta(days=1)
            )
            alert4 = Alert(
                model_id=model2.id,
                alert_type='model_drift',
                message='Sales model prediction variance increased by 12%',
                severity='high',
                status='unresolved',
                created_at=now - timedelta(minutes=30)
            )

            db.session.add_all([alert1, alert2, alert3, alert4])
            db.session.commit()

            print("[OK] Database seeded with sample data including loan model")

    app.run(debug=True)