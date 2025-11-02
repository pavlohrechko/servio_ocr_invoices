# Invoice Mapper API & CLI

Process supplier invoices (PDFs, JPGs) with **Google Cloud Vision OCR** and an **OpenAI LLM** (e.g., `gpt-4o-mini`) to map purchased items to a restaurant’s menu.

Two tools share a persistent memory file (`confirmed_mappings.json`) to improve suggestions over time:

- **Flask API (`app.py`)** — HTTP endpoint to process invoices.
- **CLI (`cli.py`)** — interactive terminal tool to review and confirm mappings.

---

## Project Structure

```
├── app.py                      # Flask API server
├── cli.py                      # Interactive CLI
├── core_mapper.py              # Shared core (OCR, LLM, models)
├── confirmed_mappings.json     # Persistent confirmed mappings (memory)
├── requirements.txt            # Python dependencies
├── uploads/                    # Temp upload dir for API
├── .env                        # Your API keys (create this)
└── README.md                   # This file
```

---

## Prerequisites

- Python 3.10+ (recommended)
- Google Cloud project with **Cloud Vision API** enabled
- OpenAI API access

---

## Setup

### 1) Create & Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure Environment (.env)

Create a `.env` file in the repository root:

```bash
# .env
OPENAI_API_KEY="sk-YourOpenAISecretKeyHere"
GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/your-gcloud-service-account-key.json"

# Optional: override default LLM
# DEFAULT_MODEL="gpt-4o-mini"
```

- **OPENAI_API_KEY**: from your OpenAI dashboard.
- **GOOGLE_APPLICATION_CREDENTIALS**: absolute path to the GCP Service Account JSON key with **Vision API** enabled.

### 4) Confirmed Mappings File

`confirmed_mappings.json` is included. Both the API and CLI read & write to it automatically.

---

## Run the Flask API

### Development

```bash
flask --app app run --port 5001
```

API base URL: `http://127.0.0.1:5001`

### Production (example)

```bash
gunicorn --workers 4 --bind 0.0.0.0:5001 "app:app"
```

---

## API Usage

### Endpoint

The API provides two main endpoints:

#### A) Process an Invoice

`POST /process-invoice`

Send a `POST` request with your invoice file.

- **Example `curl`:**
  ```bash
  curl -X POST [http://127.0.0.1:5001/process-invoice](http://127.0.0.1:5001/process-invoice) \
       -F "invoice=@/path/to/your/invoice.pdf"
  ```
- **Success Response (200 OK):**
  Returns a list of auto-confirmed items (from memory) and new suggestions that require user review.
  ```json
  {
    "status": "success",
    "auto_confirmed_items": [ ... ],
    "new_suggestions": [
      {
        "invoice_item": "Roma Tomatoes 10kg",
        "suggested_menu_dish": "Margherita Pizza",
        "notes": "..."
      }
    ]
  }
  ```

#### B) Confirm a Mapping

`POST /confirm-mapping`

Send a JSON payload to this endpoint to save a user's decision to the `confirmed_mappings.json` memory file. This should be called after your UI gets a confirmation from the user.

- **Request Body (JSON):**
  ```json
  {
    "invoice_item": "The exact item name from the invoice",
    "menu_item": "The menu dish to map to (or null for 'No Match')"
  }
  ```
- **Example `curl` (to confirm a match):**
  ```bash
  curl -X POST [http://127.0.0.1:5001/confirm-mapping](http://127.0.0.1:5001/confirm-mapping) \
       -H "Content-Type: application/json" \
       -d '{
             "invoice_item": "Roma Tomatoes 10kg",
             "menu_item": "Margherita Pizza"
           }'
  ```
- **Example `curl` (to confirm 'No Match'):**
  ```bash
  curl -X POST [http://127.0.0.1:5001/confirm-mapping](http://127.0.0.1:5001/confirm-mapping) \
       -H "Content-Type: application/json" \
       -d '{
             "invoice_item": "Cleaning Spray",
             "menu_item": null
           }'
  ```
- **Success Response (200 OK):**
  ```json
  {
    "status": "success",
    "message": "Mapping saved successfully.",
    "saved_mapping": {
      "invoice_item": "Roma Tomatoes 10kg",
      "menu_item": "Margherita Pizza"
    }
  }
  ```

#### Error Response (e.g., 400)

```json
{ "error": "No 'invoice' file part in the request." }
```

---

## Run the CLI

Use the original interactive tool:

```bash
python cli.py /path/to/your/invoice.pdf
```

The CLI will:

1. Run OCR and the LLM.
2. Show **Auto-Confirmed Mappings** (from `confirmed_mappings.json`).
3. Show **New Items Needing Review**.
4. Start an interactive loop to **[C]onfirm**, **[R]eject**, or **[S]kip** each new item.

---

## How It Works (High Level)

1. **OCR** — Extracts invoice text with Google Cloud Vision.
2. **Parsing** — Identifies line items (name, quantity, unit, notes).
3. **LLM Mapping** — Suggests the most likely menu dish (or none) per item.
4. **Memory** — On confirmation, saves the mapping to `confirmed_mappings.json` to auto-confirm similar future items.

---

## Tips & Notes

- Keep `uploads/` writeable by the API process.
- Ensure the `.env` path for `GOOGLE_APPLICATION_CREDENTIALS` is **absolute**.
- Rotate and protect keys; avoid committing `.env` and service account keys to VCS.

---

## Troubleshooting

- **`No 'invoice' file part`**: Ensure `-F "invoice=@/path/file.pdf"` and correct form field name `invoice`.
- **Auth errors**: Check `.env` values and that the Vision API is enabled.
- **Parsing quality**: OCR quality depends on input; scans with higher DPI and clear text work best.
- **Rate limits**: If you hit LLM/OCR rate limits, add retry/backoff logic or reduce concurrency.

---

## License

MIT (or your preferred license).
