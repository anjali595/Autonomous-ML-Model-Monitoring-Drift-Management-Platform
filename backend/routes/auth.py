import jwt
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from models import User
from models import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required.'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'message': 'Username already exists.'}), 409

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully.'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required.'}), 400

    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({'message': 'Invalid username or password.'}), 401

    token_expires = datetime.utcnow() + timedelta(hours=8)
    token = jwt.encode({
        'user_id': user.id,
        'username': user.username,
        'exp': token_expires
    }, current_app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({'message': 'Login successful.', 'token': token}), 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    return jsonify({'message': 'Logout successful.'}), 200

@auth_bp.route('/me', methods=['GET'])
def me():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not token:
        return jsonify({'message': 'Token is missing.'}), 401

    try:
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        user = User.query.get(payload['user_id'])
        if not user:
            raise ValueError('User not found')

        return jsonify({'id': user.id, 'username': user.username}), 200
    except Exception as e:
        return jsonify({'message': 'Token is invalid or expired.', 'error': str(e)}), 401