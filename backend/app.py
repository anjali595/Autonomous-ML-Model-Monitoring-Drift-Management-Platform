from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from datetime import datetime
from sqlalchemy import text
from routes.api import api_bp
from routes.auth import auth_bp
from models import db, Model, Dataset, Alert, MonitoringLog

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey123'  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Example database URI
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
        db.create_all()

        # Add missing status column to Alerts if the database schema is out of date.
        if db.engine.url.get_backend_name() == 'sqlite':
            with db.engine.connect() as connection:
                result = connection.execute(text("PRAGMA table_info(alerts)"))
                columns = [row[1] for row in result]
                if 'status' not in columns:
                    connection.execute(text("ALTER TABLE alerts ADD COLUMN status VARCHAR(20) DEFAULT 'unresolved'"))

        # Add sample data if database is empty
        if Model.query.first() is None:
            # Add sample models
            model1 = Model(name='Sales Forecasting', model_type='LSTM', version='1.0', baseline_accuracy=0.92)
            model2 = Model(name='Customer Churn', model_type='Random Forest', version='2.1', baseline_accuracy=0.88)
            model3 = Model(name='Fraud Detection', model_type='XGBoost', version='1.5', baseline_accuracy=0.95)
            
            # Add sample datasets
            dataset1 = Dataset(name='Q1 2026 Sales Data', description='Historical sales data for Q1 2026')
            dataset2 = Dataset(name='Customer Behavioral Data', description='Customer interaction and behavior logs')
            dataset3 = Dataset(name='Transaction Data', description='Financial transaction records')
            
            db.session.add_all([model1, model2, model3])
            db.session.add_all([dataset1, dataset2, dataset3])
            db.session.commit()
            
            # Add sample alerts
            alert1 = Alert(model_id=model1.id, alert_type='data_drift', message='Data distribution shifted significantly', severity='high', created_at=datetime.now())
            alert2 = Alert(model_id=model2.id, alert_type='model_drift', message='Model performance degraded by 5%', severity='medium', created_at=datetime.now())
            alert3 = Alert(model_id=model3.id, alert_type='data_drift', message='New feature patterns detected', severity='low', created_at=datetime.now())
            
            db.session.add_all([alert1, alert2, alert3])
            db.session.commit()
    
    app.run(debug=True)