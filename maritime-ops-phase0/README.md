# Maritime Operations Assistant - Phase 0 Validation

A standalone Python CLI tool that validates LLM-based email classification for maritime operations. This prototype processes `.eml` files, classifies them using a hybrid approach (keyword rules + Gemini LLM), and outputs structured JSON results for review.

## Overview

This tool helps validate whether LLM classification works effectively for real maritime emails before building production infrastructure. It classifies emails by vessel, category, urgency, and extracts actionable insights.

### Features

- Parse `.eml` files including attachment metadata
- Two-stage hybrid classification:
  1. Fast keyword-based rules (85%+ confidence threshold)
  2. Gemini LLM for uncertain cases
- Configurable categories, rules, and prompts via YAML
- Structured JSON output for each email
- Processing summary with statistics
- Graceful error handling

## Project Structure

```
maritime-ops-phase0/
├── emails/                   # Input folder for .eml files
├── results/                  # Output folder for JSON results
├── config.yaml               # Categories, rules, and prompts configuration
├── models.py                 # Pydantic data models
├── email_parser.py           # .eml file parsing logic
├── classifier.py             # Hybrid classification engine
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Google API key for Gemini API

### Setup Steps

1. **Clone or navigate to this directory**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up Google API key**

Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it as an environment variable:

```bash
# Linux/macOS
export GOOGLE_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set GOOGLE_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:GOOGLE_API_KEY='your-api-key-here'
```

For permanent setup, add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

## Usage

### Process All Emails in a Directory

```bash
python main.py --input ./emails --output ./results
```

### Process a Single Email File

```bash
python main.py --file ./emails/example.eml
```

### Verbose Mode (Print Each Result)

```bash
python main.py --input ./emails --output ./results --verbose
```

### Debug Mode (Detailed Logging)

```bash
python main.py --input ./emails --output ./results --debug
```

### Custom Configuration File

```bash
python main.py --input ./emails --config custom-config.yaml
```

## Output Format

For each processed email, a JSON file is created in the output directory with the following structure:

```json
{
  "parsed_email": {
    "message_id": "<message-id@example.com>",
    "subject": "Fresh water supply request - MV Ocean Trader",
    "sender": "ops@portservices.com",
    "recipients": ["operations@shipping.com"],
    "received_at": "2024-01-15T10:30:00Z",
    "body_text": "...",
    "attachments": [
      {
        "filename": "invoice.pdf",
        "content_type": "application/pdf",
        "size_bytes": 45678
      }
    ],
    "raw_path": "emails/example.eml"
  },
  "classification": {
    "vessel_name": "MV Ocean Trader",
    "category": "HUSBANDRY",
    "subcategory": "FRESH_WATER",
    "port": "Singapore",
    "dates_mentioned": ["2024-01-20"],
    "urgency": "MEDIUM",
    "summary": "Request for fresh water supply at Singapore port for MV Ocean Trader",
    "action_required": true,
    "confidence": 0.92,
    "source": "LLM"
  },
  "errors": [],
  "processing_time_seconds": 1.23
}
```

### Summary Report

After processing, a summary is printed to the console:

```
================================================================================
PROCESSING SUMMARY
================================================================================

Total emails processed: 25
Successful: 24
Failed: 1

Category Breakdown:
  HUSBANDRY/FRESH_WATER: 8
  HUSBANDRY/GARBAGE_DISPOSAL: 5
  HUSBANDRY/CASH_TO_MASTER: 3
  UNCATEGORIZED/(none): 8

Classification Source:
  LLM: 18
  RULES: 6

Urgency Distribution:
  LOW: 5
  MEDIUM: 12
  HIGH: 6
  URGENT: 1

Average Confidence: 0.87
Average Processing Time: 1.45s
================================================================================
```

## Configuration

All classification behavior is controlled via `config.yaml`. You can modify categories, keywords, prompts, and LLM settings without changing code.

### Adding New Categories

Edit the `categories` section in `config.yaml`:

```yaml
categories:
  YOUR_CATEGORY:
    description: "Description of the category"
    subcategories:
      YOUR_SUBCATEGORY:
        description: "Description of subcategory"
        keywords:
          - keyword1
          - keyword2
          - phrase with spaces
```

### Adjusting Classification Rules

Modify the `rules` section:

```yaml
rules:
  confidence_threshold: 0.85  # Minimum confidence for rule-based classification
  keyword_matching:
    case_sensitive: false
    multi_keyword_boost: 0.1  # Confidence boost per additional keyword
    base_confidence: 0.7      # Starting confidence for single keyword
```

### Customizing LLM Prompts

Edit the `prompts` section to change how the LLM analyzes emails:

```yaml
prompts:
  system_instruction: |
    Your custom system instruction here...

  classification_template: |
    Your custom prompt template here...
    Use {subject}, {sender}, {body}, {categories_list} as placeholders
```

### LLM Settings

Adjust the model and generation parameters:

```yaml
llm_settings:
  model_name: "gemini-2.0-flash-exp"  # or "gemini-1.5-pro"
  temperature: 0.1                     # Lower = more consistent
  max_tokens: 1024
  timeout: 30
```

## How It Works

### Classification Flow

1. **Parse Email**: Extract headers, body, and attachment metadata from `.eml` file
2. **Rule-Based Classification**: Match keywords against categories
   - If confidence ≥ 85%: Use rule-based result
   - If confidence < 85%: Proceed to LLM
3. **LLM Classification**: Send email to Gemini for analysis
4. **Save Results**: Write JSON output to results directory

### Hybrid Approach Benefits

- **Fast**: Most emails (with clear keywords) classified instantly via rules
- **Accurate**: Complex or ambiguous emails get LLM analysis
- **Cost-effective**: Reduces API calls by using rules first
- **Transparent**: Output shows whether rules or LLM was used

## Troubleshooting

### "GOOGLE_API_KEY not set" Error

Make sure you've exported the environment variable in your current shell session:

```bash
export GOOGLE_API_KEY='your-api-key'
python main.py --input ./emails
```

### "No .eml files found" Error

Ensure your input directory contains `.eml` files:

```bash
ls -la emails/*.eml
```

### LLM Classification Fails

- Check API key is valid
- Verify internet connection
- Check Gemini API quota/limits
- Review logs with `--debug` flag

### Encoding Errors

The parser handles various encodings gracefully. If issues persist:
- Check the `.eml` file is valid
- Try opening in an email client first
- Report encoding in error logs

## Testing the Tool

### Create a Test Email

Save this as `emails/test.eml`:

```
From: operations@portservices.com
To: vessel.ops@shipping.com
Subject: Fresh water supply required - MV Test Vessel
Date: Mon, 15 Jan 2024 10:00:00 +0000
Message-ID: <test123@example.com>

Dear Operations,

We need to arrange fresh water supply for MV Test Vessel arriving
at Singapore port on January 20th.

Required quantity: 50 tons
Estimated arrival: 20-Jan-2024 08:00

Please confirm availability and pricing.

Best regards,
Port Services Team
```

### Run Classification

```bash
python main.py --file emails/test.eml --verbose
```

Expected result: Category `HUSBANDRY`, Subcategory `FRESH_WATER`

## Next Steps (Future Phases)

This is Phase 0 validation only. Once validated, future phases will add:

- **Phase 1**: Database storage (PostgreSQL)
- **Phase 2**: Task generation and workflow
- **Phase 3**: Email monitoring and automation
- **Phase 4**: Web dashboard and API

## License

Internal tool - not for distribution.

## Support

For issues or questions, contact the development team.
