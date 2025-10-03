from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
import openai
from google.cloud import vision
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("invoice-mapper-cli")

# Set up API keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Rich Console for TUI
console = Console()

# ---------------------------------------------------------------------------
# Test Menu
# ---------------------------------------------------------------------------
RESTAURANT_MENU = [
    "Margherita Pizza", "Pepperoni Pizza", "Caesar Salad", "Greek Salad",
    "Spaghetti Carbonara", "Fettuccine Alfredo", "Tomato Soup",
    "Chicken Wings", "Garlic Bread", "Tiramisu", "Cheesecake", "Newland",
    "Браслет з силікону"
]

# його тут нема!

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
# Prompt Template for Invoice Mapping
# ---------------------------------------------------------------------------
def get_system_prompt(menu: List[str]) -> str:
    menu_str = ", ".join(f'"{item}"' for item in menu)
    
    EMPTY_SCHEMA = {
        "mapped_items": [{
            "invoice_item": "The name of the item found on the invoice (e.g., 'Roma Tomatoes 5kg')",
            "suggested_menu_dish": "The best matching dish from the menu list provided or null",
            "notes": "A brief explanation for the mapping or why no match was found."
        }]
    }

    return (
        "You are an expert system for a restaurant. Your task is to analyze text from a supplier invoice and map each purchased item to a dish on the restaurant's menu.\n\n"
        "Here is the restaurant's menu:\n"
        f"[{menu_str}]\n\n"
        "I will provide you with text blocks detected by OCR from an invoice. Identify the line items from the invoice text and map them according to the following rules:\n\n"
        "Rules (in order of importance):\n"
        "1. **PRIORITY 1: Direct Name Match:** If an invoice item's name contains the exact name of an item on the menu (e.g., invoice item 'Newland Barcode Scanner' and menu item 'Newland'), you MUST map them. This is your most important rule.\n"
        "2. **PRIORITY 2: Ingredient Mapping:** If no direct name match is found, THEN check if the invoice item is a clear ingredient for a menu dish (e.g., 'Tomatoes' for 'Margherita Pizza').\n"
        "3. **No Match:** If an item cannot be mapped by either of the above rules (e.g., 'Cleaning Supplies'), set 'suggested_menu_dish' to null.\n\n"
        "- Your output must be a valid JSON object following this exact structure, without any additional explanations outside of the JSON:\n"
        f"{json.dumps(EMPTY_SCHEMA, indent=2)}"
    )

# ---------------------------------------------------------------------------
# Google Vision OCR Helper
# ---------------------------------------------------------------------------
def google_vision_ocr(file_path: str | Path) -> OcrPayload:
    """Runs document text detection on a local PDF, JPG, or PNG file."""
    try:
        client = vision.ImageAnnotatorClient()
        path = Path(file_path)
        content = path.read_bytes()

        image = vision.Image(content=content)
        console.print(f"Sending '{path.name}' to Google Cloud Vision for OCR...", style="cyan")
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            logger.error(f"Google Vision API Error: {response.error.message}")
            sys.exit(3)

        full_text = response.full_text_annotation.text
        if not full_text:
            logger.error("Google Vision found no text in %s", file_path)
            sys.exit(3)
            
        logger.info("Google Vision extracted text successfully.")
        return OcrPayload(text_blocks=[TextBlock(text=full_text)])
    except Exception as e:
        logger.error(f"An error occurred during OCR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# LLM API Call
# ---------------------------------------------------------------------------
def _parse_response(text: str) -> InvoiceMappingResponse:
    """Safely parses the JSON response from the LLM."""
    try:
        if text.strip().startswith("```json"):
            text = text.strip()[7:-3]
            
        data = json.loads(text)
        return InvoiceMappingResponse(**data)
    except Exception as exc:
        logger.error("OpenAI returned unparsable JSON: %s", exc)
        logger.debug("Raw response: %s", text)
        sys.exit(2)

def call_openai_for_mapping(ocr: OcrPayload, model: str) -> InvoiceMappingResponse:
    """Calls the OpenAI Chat Completions API to map invoice items."""
    if not openai.api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    system_prompt = get_system_prompt(RESTAURANT_MENU)
    user_content = json.dumps(ocr.model_dump(), ensure_ascii=False)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    console.print("Asking OpenAI to map invoice items to the menu...", style="cyan")
    try:
        res = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        
        # This is the corrected line that fixes the bug from before
        response_text = res.choices[0].message.content
        
        return _parse_response(response_text)
    except Exception as e:
        logger.error(f"Failed to process OpenAI response: {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Terminal UI Display
# ---------------------------------------------------------------------------
def display_results_table(response: InvoiceMappingResponse):
    """Displays the mapping results in a formatted table in the terminal."""
    table = Table(title="Invoice to Menu Mapping Suggestions")
    table.add_column("Invoice Item", style="magenta", no_wrap=True)
    table.add_column("Suggested Menu Dish", style="green")
    table.add_column("Notes", style="yellow")

    for item in response.mapped_items:
        dish = item.suggested_menu_dish if item.suggested_menu_dish else "[dim]No Match Found[/dim]"
        table.add_row(item.invoice_item, dish, item.notes)

    console.print(table)

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Map invoice items to a restaurant menu using OCR and LLM.")
    parser.add_argument("input", help="Path to the invoice file (PDF, JPG, PNG)")
    parser.add_argument("--model", "-m", dest="model_id", default="gpt-4o-mini", help="Override OpenAI model ID")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("File not found: %s", input_path)
        sys.exit(1)

    ocr_payload = google_vision_ocr(input_path)
    mapping_response = call_openai_for_mapping(ocr_payload, model=args.model_id)
    display_results_table(mapping_response)

if __name__ == "__main__":
    main()