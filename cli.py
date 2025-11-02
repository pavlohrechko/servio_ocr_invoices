"""
Command-Line Interface (CLI) for interactive invoice mapping.
This file contains the original `rich` and `argparse` logic.
"""
from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path
from typing import List

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# Import the refactored core logic
import core_mapper
from core_mapper import (
    load_confirmed_mappings, 
    google_vision_ocr, 
    call_openai_for_mapping,
    save_confirmed_mapping,
    InvoiceMappingResponse,
    MappedItem,
    RESTAURANT_MENU
)

# ---------------------------------------------------------------------------
# CLI-specific setup
# ---------------------------------------------------------------------------
console = Console()
logger = logging.getLogger("invoice-mapper-cli")

# ---------------------------------------------------------------------------
# Terminal UI Display
# ---------------------------------------------------------------------------
def display_results_table(items: List[MappedItem], title: str, style: str = "default"):
    """Displays the mapping results in a formatted table in the terminal."""
    table = Table(title=title, style=style, title_style=style)
    table.add_column("Invoice Item", style="magenta", no_wrap=True)
    table.add_column("Suggested Menu Dish", style="green")
    table.add_column("Notes", style="yellow")
    for item in items:
        dish = item.suggested_menu_dish if item.suggested_menu_dish else "[dim]No Match Found[/dim]"
        table.add_row(item.invoice_item, dish, item.notes)
    console.print(table)

# ---------------------------------------------------------------------------
# Interactive loop for user confirmation and correction
# ---------------------------------------------------------------------------
def handle_user_confirmation_loop(items_to_review: List[MappedItem]):
    """Handles the interactive user feedback loop in the terminal for new items."""
    console.print("\n--- User Confirmation Required ---", style="bold yellow")
    console.print("For each new suggestion, please confirm, correct, or skip.")

    for item in items_to_review:
        console.print("-" * 30)
        console.print(f"Invoice Item: [magenta]{item.invoice_item}[/magenta]")
        suggested = item.suggested_menu_dish or "No Match Found"
        console.print(f"Suggested Match: [green]{suggested}[/green]")
        
        action = Prompt.ask(
            "Choose action ([C]onfirm, [R]eject/Correct, [S]kip)",
            choices=["c", "r", "s"],
            default="c",
        ).lower()

        if action == "c":
            save_confirmed_mapping(item.invoice_item, item.suggested_menu_dish)
            console.print("[bold green]✔ Mapping confirmed and saved to memory.[/bold green]")
        elif action == "r":
            console.print("Please select the correct menu item:")
            menu_options = RESTAURANT_MENU
            for i, menu_item in enumerate(menu_options):
                console.print(f"  [cyan]{i}[/cyan]: {menu_item}")
            console.print(f"  [cyan]n[/cyan]: No Match Found")

            while True:
                choice = Prompt.ask("Enter the number (or 'n' for no match)").lower()
                if choice == 'n':
                    save_confirmed_mapping(item.invoice_item, None)
                    console.print("[bold yellow]✔ Marked as 'No Match' and saved to memory.[/bold yellow]")
                    break
                try:
                    choice_idx = int(choice)
                    if 0 <= choice_idx < len(menu_options):
                        correct_menu_item = menu_options[choice_idx]
                        save_confirmed_mapping(item.invoice_item, correct_menu_item)
                        console.print(f"[bold green]✔ Corrected mapping to '{correct_menu_item}' and saved.[/bold green]")
                        break
                    else:
                        console.print("[red]Invalid number, please try again.[/red]")
                except ValueError:
                    console.print("[red]Invalid input, please enter a number or 'n'.[/red]")
        elif action == "s":
            console.print("[dim]Skipped. This mapping will not be saved.[/dim]")

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
        logger.error(f"File not found: {input_path}")
        sys.exit(1)

    try:
        confirmed_mappings = load_confirmed_mappings()
        console.print(f"Loaded {len(confirmed_mappings)} confirmed mappings from memory.", style="dim")

        console.print(f"Sending '{input_path.name}' to Google Cloud Vision for OCR...", style="cyan")
        ocr_payload = google_vision_ocr(input_path)
        
        console.print("Asking OpenAI to map invoice items to the menu (with memory)...", style="cyan")
        mapping_response = call_openai_for_mapping(
            ocr_payload, 
            model=args.model_id, 
            confirmed_mappings=confirmed_mappings
        )
        
        # --- Split items into confirmed and to-be-reviewed ---
        items_to_review = []
        auto_confirmed_items = []
        for item in mapping_response.mapped_items:
            if item.invoice_item in confirmed_mappings:
                # Override LLM suggestion with our confirmed one to ensure consistency
                item.suggested_menu_dish = confirmed_mappings[item.invoice_item]
                auto_confirmed_items.append(item)
            else:
                items_to_review.append(item)

        # Display auto-confirmed items first, if any
        if auto_confirmed_items:
            display_results_table(
                auto_confirmed_items,
                title="Auto-Confirmed Mappings (from Memory)",
                style="green"
            )

        # If there are new items, display them and start the confirmation loop
        if items_to_review:
            display_results_table(
                items_to_review,
                title="New Items Needing Review",
                style="yellow"
            )
            handle_user_confirmation_loop(items_to_review)
        elif not auto_confirmed_items:
            console.print("[yellow]No items were found on the invoice to map.[/yellow]")
        else:
            # This case runs when all items were auto-confirmed
            console.print("\n[bold green]✨ All items were auto-confirmed. No manual review needed.[/bold green]")

        console.print("\n[bold blue]Process complete.[/bold blue]")

    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
        console.print(f"\n[bold red]An error occurred:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
