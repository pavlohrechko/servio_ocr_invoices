"""
Core logic for invoice mapping.
Refactored to support multiple customers (lists and mappings).
"""
from __future__ import annotations

import json
import logging
import os
import io
from pathlib import Path
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
from pydantic import BaseModel
import openai
from google.cloud import vision
from pdf2image import convert_from_path

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("core-mapper")

# openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",  # Default Ollama URL
    api_key="ollama"  # Required string, but content doesn't matter
)

# Base directory for customer data
CUSTOMERS_DIR = Path("customers")
CUSTOMERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Customer Data Management
# ---------------------------------------------------------------------------

def get_list_path(customer_id: str) -> Path:
    """Returns path to the customer's product list."""
    return CUSTOMERS_DIR / f"{customer_id}_list.json"

def get_mappings_path(customer_id: str) -> Path:
    """Returns path to the customer's confirmed mappings."""
    return CUSTOMERS_DIR / f"{customer_id}_mappings.json"

def load_customer_list(customer_id: str) -> List[str]:
    """Loads the specific product list for a customer."""
    path = get_list_path(customer_id)
    if not path.exists():
        logger.warning(f"List file not found for customer {customer_id}.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                logger.error(f"Invalid format in {path}. Expected a JSON list.")
                return []
    except Exception as e:
        logger.error(f"Error loading list for {customer_id}: {e}")
        return []

def load_confirmed_mappings(customer_id: str) -> Dict[str, str | None]:
    """Loads the confirmed mappings for a specific customer."""
    path = get_mappings_path(customer_id)
    if not path.exists():
        # Create empty if it doesn't exist
        save_confirmed_mappings_file(customer_id, {})
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading mappings for {customer_id}: {e}")
        return {}

def save_confirmed_mappings_file(customer_id: str, mappings: Dict):
    """Helper to write the mappings file."""
    path = get_mappings_path(customer_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error saving mappings for {customer_id}: {e}")

def save_confirmed_mapping(customer_id: str, invoice_item: str, list_item: str | None):
    """Saves a single confirmed mapping to the customer's file."""
    mappings = load_confirmed_mappings(customer_id)
    mappings[invoice_item] = list_item
    save_confirmed_mappings_file(customer_id, mappings)
    logger.info(f"[{customer_id}] Saved mapping: '{invoice_item}' -> '{list_item}'")

def initialize_customer_files(customer_id: str, list_data: List[str]):
    """
    Saves the uploaded list and initializes an empty mapping file if needed.
    """
    # 1. Save the List
    list_path = get_list_path(customer_id)
    with open(list_path, "w", encoding="utf-8") as f:
        json.dump(list_data, f, indent=2, ensure_ascii=False)
    
    # 2. Init Mappings (only if not exists, to preserve history if re-uploading list)
    mappings_path = get_mappings_path(customer_id)
    if not mappings_path.exists():
        with open(mappings_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class TextBlock(BaseModel):
    text: str

class OcrPayload(BaseModel):
    text_blocks: List[TextBlock]

class MappedItem(BaseModel):
    product_code: Optional[str] = None
    invoice_item: str
    quantity: Optional[float] = None
    price: Optional[float] = None
    amount: Optional[float] = None
    suggested_item: Optional[str] = None
    notes: str = ""

class InvoiceMappingResponse(BaseModel):
    mapped_items: List[MappedItem]

# ---------------------------------------------------------------------------
# Prompt & Logic
# ---------------------------------------------------------------------------
def get_system_prompt(customer_list: List[str], confirmed_mappings: Dict[str, str | None]) -> str:
    # Convert list to string for prompt
    list_str = ", ".join(f'"{item}"' for item in customer_list)
    
    EMPTY_SCHEMA = {
        "mapped_items": [{
            "product_code": "Code or null",
            "invoice_item": "Product Name",
            "quantity": 0.0,
            "price": 0.0,
            "amount": 0.0,
            "suggested_item": "Matching item from the provided list or null",
            "notes": "VAT info etc."
        }]
    }

    confirmed_mappings_str = ""
    if confirmed_mappings:
        mappings_json = json.dumps(confirmed_mappings, indent=2, ensure_ascii=False)
        confirmed_mappings_str = (
            "You have access to previously confirmed mappings for this customer:\n"
            f"{mappings_json}\n\n"
        )

    return (
        "You are an expert procurement assistant. Analyze the supplier invoice text.\n"
        "Extract details: Product Code, Name, Quantity, Unit Price, Total Amount.\n"
        "Handle VAT: Extract net amounts (without VAT). Note VAT included prices in 'notes'.\n\n"
        
        "AFTER extracting, map the 'invoice_item' to the closest match in the provided **Customer Reference List**.\n\n"
        
        f"{confirmed_mappings_str}"
        "**Customer Reference List**:\n"
        f"[{list_str}]\n\n"
        
        "Mapping Priority:\n"
        "1. **Confirmed Mapping**: If a match exists in the provided mappings JSON, USE IT.\n"
        "2. **Direct/Fuzzy Match**: Find the best fit in the Customer Reference List.\n"
        "3. **No Match**: Set 'suggested_item' to null.\n\n"
        
        "Output strictly valid JSON:"
        f"{json.dumps(EMPTY_SCHEMA, indent=2, ensure_ascii=False)}"
    )

# ---------------------------------------------------------------------------
# OCR & LLM
# ---------------------------------------------------------------------------
def google_vision_ocr(file_path: str | Path) -> OcrPayload:
    # (Same implementation as before, omitted for brevity but assume it's here)
    # ... [Copy the OCR function from previous file here] ...
    client = vision.ImageAnnotatorClient()
    path = Path(file_path)
    full_text_combined = ""

    try:
        if path.suffix.lower() == '.pdf':
            images = convert_from_path(str(path))
            for image in images:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                content = img_byte_arr.getvalue()
                vision_image = vision.Image(content=content)
                response = client.document_text_detection(image=vision_image)
                if response.full_text_annotation.text:
                    full_text_combined += response.full_text_annotation.text + "\n"
        else:
            content = path.read_bytes()
            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)
            if response.full_text_annotation.text:
                full_text_combined = response.full_text_annotation.text

        return OcrPayload(text_blocks=[TextBlock(text=full_text_combined)])
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        raise

def call_openai_for_mapping(
    ocr: OcrPayload, 
    model: str, 
    customer_list: List[str],
    confirmed_mappings: Dict[str, str | None]
) -> InvoiceMappingResponse:
    
    # if not openai.api_key:
    #     raise ValueError("OPENAI_API_KEY not set.")

    system_prompt = get_system_prompt(customer_list, confirmed_mappings)
    user_content = json.dumps(ocr.model_dump(), ensure_ascii=False)

    # try:
    #     res = openai.chat.completions.create(
    #         model=model,
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_content}
    #         ],
    #         temperature=0,
    #         response_format={"type": "json_object"},
    #     )
    #     data = json.loads(res.choices[0].message.content)
    #     return InvoiceMappingResponse(**data)
    try:
        res = client.chat.completions.create(  # Use 'client', not 'openai'
            model="qwen2.5:14b",  # MUST match the name you pulled in Step 3
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return InvoiceMappingResponse(**data)
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        raise