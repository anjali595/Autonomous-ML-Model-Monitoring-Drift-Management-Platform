from datetime import datetime
from flask import Blueprint, jsonify, request
from models import Model, Dataset, Alert, MonitoringLog, db
import os
from werkzeug.utils import secure_filename
from services.model_loader import load_model, predict, predict_proba

try:
    import numpy as np
except ImportError:
    np = None

api_bp = Blueprint('api', __name__)

# Configure upload folder
UPLOAD_FOLDER = 'models_store'
ALLOWED_EXTENSIONS = {'pkl', 'joblib', 'h5', 'pt', 'pth', 'onnx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

api_bp = Blueprint('api', __name__)

@api_bp.route('/dashboard-stats', methods=['GET'])
def dashboard_stats():
    total_models = Model.query.count()
    total_datasets = Dataset.query.count()
    monitoring_logs = MonitoringLog.query.count()
    active_alerts = Alert.query.filter(Alert.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0)).count()
    return jsonify({
        'total_models': total_models,
        'total_datasets': total_datasets,
        'monitoring_logs': monitoring_logs,
        'active_alerts': active_alerts
    })

@api_bp.route('/models', methods=['GET', 'POST'])
def models():
    if request.method == 'GET':
        models = Model.query.all()
        return jsonify([model.to_dict() for model in models])

    data = request.get_json() or {}
    new_model = Model(
        name=data.get('name', 'Unnamed Model'),
        model_type=data.get('model_type', 'unknown'),
        version=data.get('version', '1.0'),
        baseline_accuracy=data.get('baseline_accuracy', 0.0)
    )
    db.session.add(new_model)
    db.session.commit()
    return jsonify(new_model.to_dict()), 201

@api_bp.route('/datasets', methods=['GET', 'POST'])
def datasets():
    if request.method == 'GET':
        datasets = Dataset.query.all()
        return jsonify([dataset.to_dict() for dataset in datasets])

    data = request.get_json() or {}
    new_dataset = Dataset(
        name=data.get('name', 'Unnamed Dataset'),
        description=data.get('description', '')
    )
    db.session.add(new_dataset)
    db.session.commit()
    return jsonify(new_dataset.to_dict()), 201

@api_bp.route('/alerts', methods=['GET', 'POST'])
def alerts():
    if request.method == 'GET':
        alerts = Alert.query.order_by(Alert.created_at.desc()).limit(20).all()
        return jsonify([alert.to_dict() for alert in alerts])

    data = request.get_json() or {}
    alert = Alert(
        model_id=data.get('model_id'),
        alert_type=data.get('alert_type', 'model_drift'),
        message=data.get('message', 'Drift detected'),
        severity=data.get('severity', 'medium'),
        created_at=datetime.utcnow()
    )

    db.session.add(alert)
    db.session.commit()
    return jsonify(alert.to_dict()), 201

@api_bp.route('/monitoring-logs', methods=['GET', 'POST'])
def monitoring_logs():
    if request.method == 'GET':
        logs = MonitoringLog.query.order_by(MonitoringLog.timestamp.desc()).limit(20).all()
        return jsonify([log.to_dict() for log in logs])

    data = request.get_json() or {}
    log = MonitoringLog(
        model_id=data.get('model_id'),
        dataset_id=data.get('dataset_id'),
        accuracy=data.get('accuracy'),
        data_drift_detected=data.get('data_drift_detected', False),
        model_drift_detected=data.get('model_drift_detected', False),
        psi_score=data.get('psi_score', 0.0),
        timestamp=datetime.utcnow()
    )
    db.session.add(log)
    db.session.commit()
    return jsonify(log.to_dict()), 201

@api_bp.route('/models/<int:model_id>', methods=['GET'])
def get_model(model_id):
    model = Model.query.get_or_404(model_id)
    return jsonify(model.to_dict())

@api_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    return jsonify(dataset.to_dict())

@api_bp.route('/upload-model', methods=['POST'])
def upload_model():
    """Upload a trained model file and create a model entry"""
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'message': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Get model metadata from request
        model_name = request.form.get('name', file.filename.rsplit('.', 1)[0])
        model_type = request.form.get('model_type', 'Custom')
        version = request.form.get('version', '1.0')
        baseline_accuracy = float(request.form.get('baseline_accuracy', 0.0))
        
        # Save file with secure filename
        filename = secure_filename(f"{model_name}_{version}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Create model entry in database
        new_model = Model(
            name=model_name,
            model_type=model_type,
            version=version,
            baseline_accuracy=baseline_accuracy,
            model_file_path=filepath
        )
        db.session.add(new_model)
        db.session.commit()
        
        return jsonify({
            'message': 'Model uploaded successfully',
            'model': new_model.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error uploading model: {str(e)}'}), 500

@api_bp.route('/predict/<int:model_id>', methods=['POST'])
def predict_endpoint(model_id):
    """Make predictions using a specific model"""
    try:
        model = Model.query.get_or_404(model_id)
        
        if not model.model_file_path:
            return jsonify({'message': 'Model file path not set'}), 400
        
        data = request.get_json() or {}
        input_data = data.get('features')
        
        if not input_data:
            return jsonify({'message': 'No features provided'}), 400
        
        # Convert to array if numpy is available, otherwise use list
        if np:
            input_array = np.array(input_data).reshape(1, -1)
        else:
            input_array = [input_data]
        
        # Get predictions
        predictions = predict(model.model_file_path, input_array)
        
        # Try to get probabilities
        try:
            probabilities = predict_proba(model.model_file_path, input_array)
            if np:
                proba_list = probabilities[0].tolist()
            else:
                proba_list = list(probabilities[0]) if len(probabilities) > 0 else None
        except:
            proba_list = None
        
        # Extract prediction value
        try:
            if np and hasattr(predictions[0], 'item'):
                pred_value = predictions[0].item()
            else:
                pred_value = int(predictions[0]) if isinstance(predictions[0], (int, float)) else predictions[0]
        except:
            pred_value = predictions[0]
        
        return jsonify({
            'model_id': model_id,
            'model_name': model.name,
            'prediction': pred_value,
            'probabilities': proba_list,
            'features_count': len(input_data)
        }), 200
    
    except FileNotFoundError as e:
        return jsonify({'message': f'Model file not found: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500

@api_bp.route('/models/<int:model_id>/predict', methods=['POST'])
def model_predict(model_id):
    """Alternative endpoint for predictions"""
    return predict_endpoint(model_id)

@api_bp.route('/models/<int:model_id>/info', methods=['GET'])
def model_info(model_id):
    """Get detailed model information"""
    model = Model.query.get_or_404(model_id)
    response = model.to_dict()
    
    # Add file existence check
    if model.model_file_path:
        response['file_exists'] = os.path.exists(model.model_file_path)
        response['file_size'] = os.path.getsize(model.model_file_path) if response['file_exists'] else None
    
    return jsonify(response), 200
