"""
Flask API server for the Invoice Mapper.
Provides an endpoint to upload an invoice and receive mapping suggestions.
"""
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import the refactored core logic
import core_mapper
from core_mapper import MappedItem, InvoiceMappingResponse

# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Configure a directory to save uploaded invoices
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))
app.config["UPLOADS_DIR"] = UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Ensure the uploads directory exists
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Get the logger
logger = logging.getLogger("flask-api")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.route('/')
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "message": "Invoice Mapper API is running."})

@app.route('/process-invoice', methods=['POST'])
def process_invoice():
    """
    API endpoint to process an uploaded invoice file.
    Expects a multipart/form-data request with a file part named 'invoice'.
    """
    # --- 1. File Upload Handling ---
    if 'invoice' not in request.files:
        logger.warning("API call missing 'invoice' file part.")
        return jsonify({"error": "No 'invoice' file part in the request."}), 400
    
    file = request.files['invoice']
    
    if file.filename == '':
        logger.warning("API call with no selected file.")
        return jsonify({"error": "No file selected."}), 400
        
    if not file or not allowed_file(file.filename):
        logger.warning(f"API call with disallowed file type: {file.filename}")
        return jsonify({"error": "File type not allowed. Use PDF, PNG, or JPG."}), 400

    filename = secure_filename(file.filename)
    saved_filepath = app.config["UPLOADS_DIR"] / filename
    
    try:
        file.save(saved_filepath)
        logger.info(f"File saved to {saved_filepath}")
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {e}")
        return jsonify({"error": f"Failed to save file: {e}"}), 500

    # --- 2. Core Logic Execution ---
    try:
        # Load confirmed mappings
        confirmed_mappings = core_mapper.load_confirmed_mappings()
        logger.info(f"Loaded {len(confirmed_mappings)} confirmed mappings.")
        
        # Perform OCR
        ocr_payload = core_mapper.google_vision_ocr(saved_filepath)
        
        # Call LLM for mapping
        model_id = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        mapping_response = core_mapper.call_openai_for_mapping(
            ocr_payload, 
            model=model_id,
            confirmed_mappings=confirmed_mappings
        )
        
        # --- 3. Process and Return Results ---
        # Split items into confirmed and to-be-reviewed
        items_to_review = []
        auto_confirmed_items = []
        for item in mapping_response.mapped_items:
            if item.invoice_item in confirmed_mappings:
                # Override LLM suggestion with our confirmed one
                item.suggested_menu_dish = confirmed_mappings[item.invoice_item]
                auto_confirmed_items.append(item)
            else:
                items_to_review.append(item)
        
        logger.info(f"Processing complete. Found {len(auto_confirmed_items)} auto-confirmed and {len(items_to_review)} new items.")
        
        # Return the structured JSON response
        return jsonify({
            "status": "success",
            "auto_confirmed_items": [item.model_dump() for item in auto_confirmed_items],
            "new_suggestions": [item.model_dump() for item in items_to_review]
        }), 200

    except Exception as e:
        # Catch errors from OCR or LLM
        logger.error(f"Error processing file {filename}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        try:
            os.remove(saved_filepath)
            logger.info(f"Cleaned up file: {saved_filepath}")
        except OSError as e:
            logger.error(f"Error deleting file {saved_filepath}: {e}")

@app.route('/confirm-mapping', methods=['POST'])
def confirm_mapping():
    """
    API endpoint to save a user-confirmed or corrected mapping to the database.
    Expects a JSON payload like:
    {
        "invoice_item": "Name of Item from Invoice",
        "menu_item": "Name of Menu Dish"  (or null for 'No Match')
    }
    """
    data = request.json
    
    if not data or 'invoice_item' not in data:
        logger.warning("Confirmation call missing 'invoice_item' in JSON payload.")
        return jsonify({"error": "Missing 'invoice_item' in JSON payload."}), 400

    invoice_item = data.get('invoice_item')
    # .get('menu_item') will default to None if the key is missing or set to null
    menu_item = data.get('menu_item') 
    
    try:
        core_mapper.save_confirmed_mapping(invoice_item, menu_item)
        logger.info(f"Saved confirmed mapping for: '{invoice_item}' -> '{menu_item}'")
        return jsonify({
            "status": "success", 
            "message": "Mapping saved successfully.",
            "saved_mapping": { "invoice_item": invoice_item, "menu_item": menu_item }
        }), 200
    except Exception as e:
        logger.error(f"Failed to save mapping for '{invoice_item}': {e}")
        return jsonify({"error": f"Failed to save mapping: {e}"}), 500

# ---------------------------------------------------------------------------
# Run the App
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Note: Use `flask --app app run` in development,
    # or a Gunicorn server in production.
    app.run(debug=True, host='0.0.0.0', port=5001)
