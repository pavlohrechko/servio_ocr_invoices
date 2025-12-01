"""
Flask API server for Invoice Mapper (Multi-Customer Support).
"""
import os
import logging
import json
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import refactored core logic
import core_mapper

app = Flask(__name__)

# Config
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
app.config["UPLOADS_DIR"] = UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Increased to 32MB
ALLOWED_INVOICE_EXTS = {'pdf', 'png', 'jpg', 'jpeg'}
ALLOWED_LIST_EXTS = {'json'}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("flask-api")

def allowed_file(filename, extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def health_check():
    return jsonify({"status": "ok", "message": "Multi-Tenant Invoice Mapper API is running."})

# ---------------------------------------------------------------------------
# 1. UPLOAD CUSTOMER LIST
# ---------------------------------------------------------------------------
@app.route('/upload-list', methods=['POST'])
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
        content = json.load(file)
        
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
def process_invoice():
    """
    Process an invoice using the customer's specific context.
    Form Data:
      - customer_id (string)
      - invoice (file)
    """
    customer_id = request.form.get('customer_id')
    if not customer_id:
        return jsonify({"error": "Missing 'customer_id' in form data."}), 400

    if 'invoice' not in request.files:
        return jsonify({"error": "No 'invoice' file part."}), 400
    
    file = request.files['invoice']
    if not file or not allowed_file(file.filename, ALLOWED_INVOICE_EXTS):
        return jsonify({"error": "Invalid invoice file type."}), 400

    # Check if customer list exists
    customer_list = core_mapper.load_customer_list(customer_id)
    if not customer_list:
        return jsonify({
            "error": f"No list found for customer '{customer_id}'. Please call /upload-list first."
        }), 404

    filename = secure_filename(file.filename)
    saved_filepath = app.config["UPLOADS_DIR"] / f"{customer_id}_{filename}"
    
    try:
        file.save(saved_filepath)
        
        # Load mappings specific to this customer
        confirmed_mappings = core_mapper.load_confirmed_mappings(customer_id)
        
        # 1. OCR
        ocr_payload = core_mapper.google_vision_ocr(saved_filepath)
        
        # 2. LLM with Customer Context
        model_id = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        mapping_response = core_mapper.call_openai_for_mapping(
            ocr_payload, 
            model=model_id,
            customer_list=customer_list,
            confirmed_mappings=confirmed_mappings
        )
        
        # 3. Filter Results
        items_to_review = []
        auto_confirmed_items = []
        
        for item in mapping_response.mapped_items:
            # Check strict match against confirmed mappings
            if item.invoice_item in confirmed_mappings:
                item.suggested_item = confirmed_mappings[item.invoice_item]
                auto_confirmed_items.append(item)
            else:
                items_to_review.append(item)
        
        return jsonify({
            "status": "success",
            "customer_id": customer_id,
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
    app.run(debug=True, host='0.0.0.0', port=5000)