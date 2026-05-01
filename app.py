"""
Flask API server for Invoice Mapper (Multi-Customer Support).
"""
import os
import logging
import json
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd 
import time

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from functools import wraps

# Import refactored core logic
import core_mapper

app = Flask(__name__)

# Config
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
app.config["UPLOADS_DIR"] = UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Increased to 32MB

ALLOWED_INVOICE_EXTS = {'pdf', 'png', 'jpg', 'jpeg', 'heic', 'csv', 'xlsx', 'xls', 'docx', 'doc'}
ALLOWED_LIST_EXTS = {'json'}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("flask-api")

def normalize(s: str) -> str:
    return s.strip().lower()

def allowed_file(filename, extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        
        token = auth_header.split(" ")[1]
        
        try:
            id_info = id_token.verify_oauth2_token(
                token,
                google_requests.Request(),
                GOOGLE_CLIENT_ID
            )
            request.user = {
                "email": id_info.get("email"),
                "name": id_info.get("name"),
                "sub": id_info.get("sub")
            }
        except ValueError as e:
            return jsonify({"error": f"Invalid token: {e}"}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@require_auth
def health_check():
    return jsonify({"status": "ok", "message": "Multi-Tenant Invoice Mapper API is running."})

# ---------------------------------------------------------------------------
# 1. UPLOAD CUSTOMER LIST
# ---------------------------------------------------------------------------
@app.route('/upload-list', methods=['POST'])
@require_auth
def upload_list():
    """
    Endpoint to upload a customer's specific item list .
    Form Data:
      - customer_id (string): Unique identifier for the customer
      - file (file): A JSON file containing a list of strings
    """
    # Check customer_id
    customer_id = request.form.get('customer_id')
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in form data."}), 400

    # Check file
    if 'file' not in request.files:
        return jsonify({"error": "No file part."}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename, ALLOWED_LIST_EXTS):
        return jsonify({"error": "Invalid file. Upload a JSON file."}), 400

    try:
        # Parse JSON
        content = json.loads(file.read().decode('utf-8-sig'))
        
        # Validation: Must be a list of strings
        if not isinstance(content, list):
             return jsonify({"error": "JSON root must be a list."}), 400
        
        # Save list and init mappings using core logic
        core_mapper.initialize_customer_files(customer_id, content)
        
        logger.info(f"List uploaded for customer {customer_id} ({len(content)} items).")
        return jsonify({
            "status": "success",
            "message": f"List for '{customer_id}' saved successfully.",
            "item_count": len(content)
        }), 200

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format."}), 400
    except Exception as e:
        logger.error(f"Error uploading list for {customer_id}: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# 2. PROCESS INVOICE
# ---------------------------------------------------------------------------

@app.route('/process-invoice', methods=['POST'])
@require_auth
def process_invoice():
    start_time = time.time()
    customer_id = request.form.get('customer_id')
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in form data."}), 400

    if 'invoice' not in request.files:
        return jsonify({"error": "No 'invoice' file part."}), 400

    file = request.files['invoice']
    if not file or not allowed_file(file.filename, ALLOWED_INVOICE_EXTS):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_INVOICE_EXTS}"}), 400

    customer_list = core_mapper.load_customer_list(customer_id)
    if not customer_list:
        return jsonify({"error": f"No list found for customer '{customer_id}'."}), 404

    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    saved_filepath = app.config["UPLOADS_DIR"] / f"{customer_id}_{filename}"

    try:
        file.save(saved_filepath)
        app.json.ensure_ascii = False
        confirmed_mappings = core_mapper.load_confirmed_mappings(customer_id)

        mapping_response = core_mapper.call_gemini_for_mapping(
            saved_filepath,
            ext=ext,
            model="gemini-2.5-flash",
            customer_list=customer_list,
            confirmed_mappings=confirmed_mappings
        )

        items_to_review = []
        auto_confirmed_items = []
        confirmed_lower = {normalize(k): v for k, v in confirmed_mappings.items()}

        for item in mapping_response.mapped_items:
            key = normalize(item.invoice_item)
            if key in confirmed_lower:
                item.suggested_item = confirmed_lower[key]
                auto_confirmed_items.append(item)
            else:
                items_to_review.append(item)

        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            "status": "success",
            "customer_id": customer_id,
            "processing_time_sec": elapsed,
            "auto_confirmed_items": [item.model_dump() for item in auto_confirmed_items],
            "new_suggestions": [item.model_dump() for item in items_to_review]
        }), 200

    except Exception as e:
        logger.error(f"Error processing invoice: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(saved_filepath):
                os.remove(saved_filepath)
        except OSError:
            pass

# ---------------------------------------------------------------------------
# 3. CONFIRM MAPPING
# ---------------------------------------------------------------------------
@app.route('/confirm-mapping', methods=['POST'])
@require_auth
def confirm_mapping():
    """
    Confirm a mapping for a specific customer.
    JSON Body:
      {
        "customer_id": "123",
        "invoice_item": "Raw Item Name",
        "list_item": "Mapped List Item" (or null)
      }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
        
    customer_id = data.get('customer_id')
    invoice_item = data.get('invoice_item')
    list_item = data.get('list_item')
    
    if not customer_id or not invoice_item:
        return jsonify({"error": "Missing 'customer_id' or 'invoice_item'."}), 400
    
    try:
        core_mapper.save_confirmed_mapping(customer_id, invoice_item, list_item)
        return jsonify({
            "status": "success", 
            "message": "Mapping saved.",
            "customer_id": customer_id,
            "mapping": { "invoice_item": invoice_item, "list_item": list_item }
        }), 200
    except Exception as e:
        logger.error(f"Failed to save mapping: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)