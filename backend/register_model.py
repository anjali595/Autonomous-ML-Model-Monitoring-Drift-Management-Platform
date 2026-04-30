#!/usr/bin/env python
"""
Script to register a pre-trained model in the database.
Usage: python register_model.py <model_file_path> <model_name> <model_type> <version> <baseline_accuracy>

Example:
  python register_model.py models_store/loan_model.pkl "Loan Classifier" "Random Forest" "1.0" "0.87"
"""

import sys
import os
from app import app, db
from models import Model
from datetime import datetime

def register_model(file_path, name, model_type, version, baseline_accuracy):
    """Register a model in the database"""
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return False
    
    try:
        with app.app_context():
            # Check if model already exists
            existing = Model.query.filter_by(name=name, version=version).first()
            if existing:
                print(f"⚠️  Model already exists: {name} v{version}")
                return False
            
            # Create new model entry
            model = Model(
                name=name,
                model_type=model_type,
                version=version,
                baseline_accuracy=float(baseline_accuracy),
                model_file_path=file_path,
                created_at=datetime.utcnow()
            )
            
            db.session.add(model)
            db.session.commit()
            
            print(f"✅ Model registered successfully!")
            print(f"   ID: {model.id}")
            print(f"   Name: {model.name}")
            print(f"   Type: {model.model_type}")
            print(f"   Version: {model.version}")
            print(f"   Accuracy: {model.baseline_accuracy}")
            print(f"   File: {model.model_file_path}")
            
            return True
    
    except Exception as e:
        print(f"❌ Error registering model: {str(e)}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: python register_model.py <model_file_path> <name> <type> <version> <baseline_accuracy>")
        print("\nExample:")
        print('  python register_model.py models_store/loan_model.pkl "Loan Classifier" "Random Forest" "1.0" "0.87"')
        sys.exit(1)
    
    file_path = sys.argv[1]
    name = sys.argv[2]
    model_type = sys.argv[3]
    version = sys.argv[4]
    baseline_accuracy = sys.argv[5]
    
    success = register_model(file_path, name, model_type, version, baseline_accuracy)
    sys.exit(0 if success else 1)
