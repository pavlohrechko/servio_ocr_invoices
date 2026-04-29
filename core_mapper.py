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
load_dotenv()  # Must be before anything reads env vars

from google import genai
from pydantic import BaseModel
from pdf2image import convert_from_path

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=GENAI_API_KEY)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("core-mapper")

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
    path = get_list_path(customer_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                logger.error(f"Invalid format in {path}. Expected a JSON list.")
                return []
    except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
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
    except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
        logger.error(f"Error loading mappings for {customer_id}: {e}")
        save_confirmed_mappings_file(customer_id, {})
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
    unit: Optional[str] = None
    price: Optional[float] = None
    price_nds: Optional[float] = None
    amount: Optional[float] = None
    amount_nds: Optional[float] = None
    nds: Optional[str] = None
    suggested_item: Optional[str] = None
    notes: Optional[str] = ""
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
            "unit": "Unit of measurement or null",
            "price": 0.0,
            "price_nds": 0.0,
            "amount": 0.0,
            "amount_nds": 0.0,
            "nds": "VAT percentage as string e.g. '20%' or null",
            "suggested_item": "Matching item from the provided list or null",
            "notes": "Any additional info"
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
        "Extract details: Product Code, Name, Quantity, Unit of Measurement, "
        "Unit Price without VAT (price), Unit Price with VAT (price_nds), "
        "Total Amount without VAT (amount), Total Amount with VAT (amount_nds), "
        "VAT percentage as string (nds, e.g. '20%', '14%').\n"
        "If only one price is given, derive the other using the VAT rate found in the document.\n\n"

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
# LLM
# ---------------------------------------------------------------------------

def call_gemini_for_mapping(image_path_or_ocr, model, customer_list, confirmed_mappings, is_excel=False):
    system_prompt = get_system_prompt(customer_list, confirmed_mappings)

    if is_excel:
        # Excel: already text, send as string
        user_content = image_path_or_ocr
        contents = [system_prompt + "\n\n" + user_content]
    else:
        # Image/PDF: send directly to Gemini Vision
        path = Path(image_path_or_ocr)
        if path.suffix.lower() == '.pdf':
            from pdf2image import convert_from_path
            import PIL.Image
            images = convert_from_path(str(path))
            contents = [system_prompt] + images
        else:
            import PIL.Image
            image = PIL.Image.open(path)
            contents = [system_prompt, image]

    response = genai_client.models.generate_content(
        model=model,
        contents=contents
    )
    text = response.text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    logger.info(f"Gemini raw response: {text[:500]}")

    data = json.loads(text)

    items = data.get("mapped_items", [])
    if not isinstance(items, list):
        raise ValueError(f"Unexpected mapped_items format: {type(items)}")

    sanitized_items = []
    for item in items:
        if isinstance(item, dict):
            if item.get("suggested_item") == "null":
                item["suggested_item"] = None
            sanitized_items.append(item)
        else:
            logger.warning(f"Skipping non-dict item: {item}")

    return InvoiceMappingResponse(mapped_items=sanitized_items)