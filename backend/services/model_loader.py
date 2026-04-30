import pickle
import joblib
import os

try:
    import numpy as np
except ImportError:
    np = None

def load_model(model_file_path):
    """Load a trained model from file"""
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    
    if model_file_path.endswith('.pkl'):
        with open(model_file_path, 'rb') as f:
            return pickle.load(f)
    elif model_file_path.endswith('.joblib'):
        return joblib.load(model_file_path)
    else:
        raise ValueError(f"Unsupported format: {model_file_path}")

def predict(model_file_path, input_data):
    """Make predictions using loaded model"""
    model = load_model(model_file_path)
    return model.predict(input_data)

def predict_proba(model_file_path, input_data):
    """Get prediction probabilities"""
    model = load_model(model_file_path)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(input_data)
    else:
        raise ValueError("Model does not support predict_proba")
