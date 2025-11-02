"""
Core logic for invoice mapping, refactored for import by API or CLI.
Based on the user's original mapping.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from pydantic import BaseModel
import openai
from google.cloud import vision

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("core-mapper")

# Set up API keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the path for our persistent memory (database)
DB_FILE = Path("confirmed_mappings.json")

# ---------------------------------------------------------------------------
# Test Menu
# ---------------------------------------------------------------------------
RESTAURANT_MENU = [
    "Margherita Pizza", "Pepperoni Pizza", "Caesar Salad", "Greek Salad",
    "Spaghetti Carbonara", "Fettuccine Alfredo", "Tomato Soup",
    "Chicken Wings", "Garlic Bread", "Tiramisu", "Cheesecake", "Newland",
    "Браслет з силікону"
]

# ---------------------------------------------------------------------------
# Pydantic Schemas for Invoice Processing
# ---------------------------------------------------------------------------
class TextBlock(BaseModel):
    text: str

class OcrPayload(BaseModel):
    text_blocks: List[TextBlock]

class MappedItem(BaseModel):
    invoice_item: str
    suggested_menu_dish: str | None
    notes: str = ""

class InvoiceMappingResponse(BaseModel):
    mapped_items: List[MappedItem]

# ---------------------------------------------------------------------------
# Functions to manage the persistent memory
# ---------------------------------------------------------------------------
def load_confirmed_mappings() -> Dict[str, str | None]:
    """Loads the confirmed mappings from the JSON database file."""
    if not DB_FILE.exists():
        return {}
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading confirmed mappings from {DB_FILE}: {e}")
        return {}

def save_confirmed_mapping(invoice_item: str, menu_item: str | None):
    """Saves a single confirmed or corrected mapping to the database."""
    mappings = load_confirmed_mappings()
    mappings[invoice_item] = menu_item
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved mapping: '{invoice_item}' -> '{menu_item}'")
    except IOError as e:
        logger.error(f"Error saving confirmed mappings to {DB_FILE}: {e}")

# ---------------------------------------------------------------------------
# Prompt Template for Invoice Mapping
# ---------------------------------------------------------------------------
def get_system_prompt(menu: List[str], confirmed_mappings: Dict[str, str | None]) -> str:
    menu_str = ", ".join(f'"{item}"' for item in menu)
    
    EMPTY_SCHEMA = {
        "mapped_items": [{
            "invoice_item": "The name of the item found on the invoice (e.g., 'Roma Tomatoes 5kg')",
            "suggested_menu_dish": "The best matching dish from the menu list provided or null",
            "notes": "A brief explanation for the mapping or why no match was found."
        }]
    }

    confirmed_mappings_str = ""
    if confirmed_mappings:
        mappings_json = json.dumps(confirmed_mappings, indent=2, ensure_ascii=False)
        confirmed_mappings_str = (
            "You have access to a memory of previously confirmed mappings. Use this information to guide your decisions.\n"
            "Here are the confirmed mappings:\n"
            f"{mappings_json}\n\n"
        )

    return (
        "You are an expert system for a restaurant. Your task is to analyze text from a supplier invoice and map each purchased item to a dish on the restaurant's menu.\n\n"
        f"{confirmed_mappings_str}"
        "Here is the restaurant's menu:\n"
        f"[{menu_str}]\n\n"
        "I will provide you with text blocks detected by OCR from an invoice. Identify the line items from the invoice text and map them according to the following rules:\n\n"
        "Rules (in order of importance):\n"
        "1. **PRIORITY 0: CONFIRMED MAPPINGS:** If an invoice item is identical or very similar to one from the confirmed mappings memory I provided, you MUST use that mapping. This is your most important rule.\n"
        "2. **PRIORITY 1: Direct Name Match:** If no direct name match, if an invoice item's name contains the exact name of an item on the menu (e.g., invoice item 'Newland Barcode Scanner' and menu item 'Newland'), you MUST map them.\n"
        "3. **PRIORITY 2: Ingredient Mapping:** If no direct name match is found, THEN check if the invoice item is a clear ingredient for a menu dish (e.g., 'Tomatoes' for 'Margherita Pizza').\n"
        "4. **No Match:** If an item cannot be mapped by any of the above rules (e.g., 'Cleaning Supplies'), set 'suggested_menu_dish' to null.\n\n"
        "- Your output must be a valid JSON object following this exact structure, without any additional explanations outside of the JSON:\n"
        f"{json.dumps(EMPTY_SCHEMA, indent=2, ensure_ascii=False)}"
    )

# ---------------------------------------------------------------------------
# Google Vision OCR Helper
# ---------------------------------------------------------------------------
def google_vision_ocr(file_path: str | Path) -> OcrPayload:
    """
    Runs document text detection on a local PDF, JPG, or PNG file.
    Raises:
        Exception: If Google Vision API returns an error or finds no text.
    """
    try:
        client = vision.ImageAnnotatorClient()
        path = Path(file_path)
        content = path.read_bytes()
        image = vision.Image(content=content)
        
        logger.info(f"Sending '{path.name}' to Google Cloud Vision for OCR...")
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            logger.error(f"Google Vision API Error: {response.error.message}")
            raise Exception(f"Google Vision API Error: {response.error.message}")

        full_text = response.full_text_annotation.text
        if not full_text:
            logger.error(f"Google Vision found no text in {file_path}")
            raise Exception(f"Google Vision found no text in {file_path}")
            
        logger.info("Google Vision extracted text successfully.")
        return OcrPayload(text_blocks=[TextBlock(text=full_text)])
    except Exception as e:
        logger.error(f"An error occurred during OCR: {e}")
        # Re-raise the exception to be caught by the caller (API or CLI)
        raise

# ---------------------------------------------------------------------------
# LLM API Call
# ---------------------------------------------------------------------------
def _parse_response(text: str) -> InvoiceMappingResponse:
    """
    Safely parses the JSON response from the LLM.
    Raises:
        json.JSONDecodeError: If the response is not valid JSON.
        pydantic.ValidationError: If JSON structure doesn't match Pydantic model.
    """
    try:
        if text.strip().startswith("```json"):
            text = text.strip()[7:-3]
        data = json.loads(text)
        return InvoiceMappingResponse(**data)
    except Exception as exc:
        logger.error(f"LLM returned unparsable JSON: {exc}")
        logger.debug(f"Raw response: {text}")
        raise  # Re-raise to be caught by caller

def call_openai_for_mapping(
    ocr: OcrPayload, model: str, confirmed_mappings: Dict[str, str | None]
) -> InvoiceMappingResponse:
    """
    Calls the OpenAI Chat Completions API to map invoice items.
    Raises:
        Exception: If the API key is not set or the API call fails.
    """
    if not openai.api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    system_prompt = get_system_prompt(RESTAURANT_MENU, confirmed_mappings)
    user_content = json.dumps(ocr.model_dump(), ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    logger.info("Asking OpenAI to map invoice items to the menu (with memory)...")
    try:
        res = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        response_text = res.choices[0].message.content
        return _parse_response(response_text)
    except Exception as e:
        logger.error(f"Failed to process OpenAI response: {e}")
        raise  # Re-raise to be caught by caller
