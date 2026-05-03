from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity
)
import bcrypt
import json
from pathlib import Path

auth_bp = Blueprint('auth', __name__)

USERS_FILE = Path("users.json")

def load_users():
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text())

def save_users(users):
    USERS_FILE.write_text(json.dumps(users, indent=2))

# ---------------------------------------------------------------------------
# Token endpoint (OAuth2 password grant)
# ---------------------------------------------------------------------------
@auth_bp.route('/oauth/token', methods=['POST'])
def token():
    """
    OAuth2 Resource Owner Password Credentials Grant
    Body (form or JSON):
      - grant_type: "password"
      - username: "user@example.com"
      - password: "secret"
    Returns:
      - access_token
      - refresh_token
      - token_type: "bearer"
    """
    data = request.get_json() or request.form

    grant_type = data.get('grant_type')
    if grant_type not in ('password', 'refresh_token'):
        return jsonify({"error": "unsupported_grant_type"}), 400

    # Refresh token grant
    if grant_type == 'refresh_token':
        return _handle_refresh(data)

    # Password grant
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "missing_credentials"}), 400

    users = load_users()
    user = users.get(username)

    if not user or not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
        return jsonify({"error": "invalid_credentials"}), 401

    access_token = create_access_token(identity=username)
    refresh_token = create_refresh_token(identity=username)

    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 3600
    }), 200


def _handle_refresh(data):
    from flask_jwt_extended import decode_token
    from flask_jwt_extended.exceptions import JWTDecodeError
    
    refresh_token = data.get('refresh_token')
    if not refresh_token:
        return jsonify({"error": "missing_refresh_token"}), 400
    
    try:
        decoded = decode_token(refresh_token)
        identity = decoded['sub']
        access_token = create_access_token(identity=identity)
        return jsonify({
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600
        }), 200
    except Exception:
        return jsonify({"error": "invalid_refresh_token"}), 401


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------
@auth_bp.route('/oauth/register', methods=['POST'])
def register():
    """
    Register a new user.
    Body: { "username": "...", "password": "..." }
    """
    data = request.get_json()
    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "missing_fields"}), 400

    users = load_users()
    if username in users:
        return jsonify({"error": "user_already_exists"}), 409

    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {"password_hash": password_hash}
    save_users(users)

    return jsonify({"status": "success", "message": f"User '{username}' created."}), 201


@auth_bp.route('/oauth/me', methods=['GET'])
@jwt_required()
def me():
    return jsonify({"username": get_jwt_identity()}), 200