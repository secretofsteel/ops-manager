# Quick Start Guide

Get up and running with the Maritime Operations Email Classifier in 5 minutes.

## Step 1: Install Dependencies

```bash
cd maritime-ops-phase0
pip install -r requirements.txt
```

## Step 2: Set Up API Key

Get a free Google Gemini API key from: https://aistudio.google.com/app/apikey

Then set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

## Step 3: Test with Sample Emails

We've included 3 sample emails to test the classifier:

```bash
python main.py --input ./emails --output ./results --verbose
```

You should see output like:

```
2024-01-20 12:00:00 - __main__ - INFO - Found 3 email file(s) to process
2024-01-20 12:00:01 - __main__ - INFO - Parsing sample_fresh_water.eml...
2024-01-20 12:00:01 - __main__ - INFO - Classifying sample_fresh_water.eml...

================================================================================
File: sample_fresh_water.eml
Subject: Fresh water supply required - MV Ocean Trader
Sender: operations@portservices.com

Category: HUSBANDRY
Subcategory: FRESH_WATER
Vessel: MV Ocean Trader
Port: Singapore
Urgency: MEDIUM
Action Required: True
Confidence: 0.92
Source: LLM

Summary: Request for fresh water supply at Singapore port for MV Ocean Trader
Processing Time: 1.23s
================================================================================
```

## Step 4: Check Results

Results are saved as JSON files in `./results/`:

```bash
ls -la results/
cat results/sample_fresh_water.json
```

## Step 5: Try Your Own Emails

1. Copy your `.eml` files to the `emails/` directory
2. Run the classifier:

```bash
python main.py --input ./emails --output ./results
```

## Expected Results for Sample Emails

| Email File | Expected Category | Expected Subcategory |
|------------|------------------|----------------------|
| sample_fresh_water.eml | HUSBANDRY | FRESH_WATER |
| sample_garbage.eml | HUSBANDRY | GARBAGE_DISPOSAL |
| sample_cash.eml | HUSBANDRY | CASH_TO_MASTER |

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Customize categories and prompts in [config.yaml](config.yaml)
- Process your real maritime operations emails
- Review and validate the classification results

## Troubleshooting

**"GOOGLE_API_KEY not set" error?**
```bash
# Make sure you've exported it in your current shell
echo $GOOGLE_API_KEY
```

**No output or errors?**
```bash
# Run with debug mode for detailed logs
python main.py --input ./emails --debug
```

**Want to test a single email?**
```bash
python main.py --file ./emails/sample_fresh_water.eml --verbose
```
