from functools import wraps
from flask import request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

AUTH_TOKEN = os.getenv("BEARER_TOKEN")

def verify_token(f):
    """Decorator to enforce Bearer token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ")[1]

        if token != AUTH_TOKEN:
            return jsonify({"error": "Invalid or missing authentication token"}), 401

        return f(*args, **kwargs)
    return decorated
