#!/usr/bin/env python3
"""
Maritime Operations Email Classification CLI

Process .eml files and classify them using hybrid rule-based and LLM approach.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import yaml

from models import ProcessedEmail
from email_parser import parse_eml_file
from classifier import EmailClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def process_email_file(
    email_path: Path,
    classifier: EmailClassifier,
    output_dir: Path,
    verbose: bool = False
) -> ProcessedEmail:
    """
    Process a single email file.

    Args:
        email_path: Path to .eml file
        classifier: Email classifier instance
        output_dir: Directory to save results
        verbose: Print results to console

    Returns:
        ProcessedEmail object
    """
    start_time = time.time()
    errors = []

    try:
        # Parse email
        logger.info(f"Parsing {email_path.name}...")
        parsed_email = parse_eml_file(email_path)

        # Classify email
        logger.info(f"Classifying {email_path.name}...")
        classification = classifier.classify(parsed_email)

        processing_time = time.time() - start_time

        result = ProcessedEmail(
            parsed_email=parsed_email,
            classification=classification,
            errors=errors,
            processing_time_seconds=processing_time
        )

        # Save result to JSON
        output_file = output_dir / f"{email_path.stem}.json"
        with open(output_file, 'w') as f:
            json.dump(result.model_dump(mode='json'), f, indent=2, default=str)

        logger.info(f"Saved result to {output_file}")

        if verbose:
            print_result(result)

        return result

    except Exception as e:
        logger.error(f"Failed to process {email_path.name}: {e}")
        errors.append(str(e))

        # Create error result
        try:
            parsed_email = parse_eml_file(email_path)
        except Exception:
            # If parsing failed, create minimal parsed email
            from models import ParsedEmail as PE
            parsed_email = PE(
                subject="(Failed to parse)",
                sender="(Unknown)",
                raw_path=email_path
            )

        processing_time = time.time() - start_time

        result = ProcessedEmail(
            parsed_email=parsed_email,
            classification=None,
            errors=errors,
            processing_time_seconds=processing_time
        )

        # Save error result
        output_file = output_dir / f"{email_path.stem}.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(result.model_dump(mode='json'), f, indent=2, default=str)
        except Exception as save_error:
            logger.error(f"Failed to save error result: {save_error}")

        return result


def print_result(result: ProcessedEmail):
    """Print a formatted result to console."""
    print("\n" + "=" * 80)
    print(f"File: {result.parsed_email.raw_path.name}")
    print(f"Subject: {result.parsed_email.subject}")
    print(f"Sender: {result.parsed_email.sender}")

    if result.classification:
        cls = result.classification
        print(f"\nCategory: {cls.category}")
        if cls.subcategory:
            print(f"Subcategory: {cls.subcategory}")
        if cls.vessel_name:
            print(f"Vessel: {cls.vessel_name}")
        if cls.port:
            print(f"Port: {cls.port}")
        print(f"Urgency: {cls.urgency.value}")
        print(f"Action Required: {cls.action_required}")
        print(f"Confidence: {cls.confidence:.2f}")
        print(f"Source: {cls.source.value}")
        print(f"\nSummary: {cls.summary}")
    else:
        print("\nClassification: FAILED")

    if result.errors:
        print(f"\nErrors: {', '.join(result.errors)}")

    print(f"Processing Time: {result.processing_time_seconds:.2f}s")
    print("=" * 80)


def print_summary(results: List[ProcessedEmail]):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    print(f"\nTotal emails processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        # Category breakdown
        print("\nCategory Breakdown:")
        category_counts = {}
        for result in results:
            if result.classification:
                cat = result.classification.category
                subcat = result.classification.subcategory or "(none)"
                key = f"{cat}/{subcat}"
                category_counts[key] = category_counts.get(key, 0) + 1

        for key, count in sorted(category_counts.items()):
            print(f"  {key}: {count}")

        # Source breakdown
        print("\nClassification Source:")
        source_counts = {}
        for result in results:
            if result.classification:
                source = result.classification.source.value
                source_counts[source] = source_counts.get(source, 0) + 1

        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count}")

        # Urgency breakdown
        print("\nUrgency Distribution:")
        urgency_counts = {}
        for result in results:
            if result.classification:
                urgency = result.classification.urgency.value
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1

        for urgency in ['LOW', 'MEDIUM', 'HIGH', 'URGENT']:
            count = urgency_counts.get(urgency, 0)
            if count > 0:
                print(f"  {urgency}: {count}")

        # Average confidence
        avg_conf = sum(r.classification.confidence for r in results if r.classification) / successful
        print(f"\nAverage Confidence: {avg_conf:.2f}")

        # Average processing time
        avg_time = sum(r.processing_time_seconds or 0 for r in results) / total
        print(f"Average Processing Time: {avg_time:.2f}s")

    print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Maritime Operations Email Classification Tool'
    )

    # Input/output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=Path,
        help='Directory containing .eml files to process'
    )
    input_group.add_argument(
        '--file',
        type=Path,
        help='Single .eml file to process'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results'),
        help='Output directory for JSON results (default: results/)'
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to configuration file (default: config.yaml)'
    )

    # Behavior arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print results to console as they are processed'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Initialize classifier
    logger.info("Initializing classifier...")
    classifier = EmailClassifier(config)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Collect email files to process
    if args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        email_files = [args.file]
    else:
        if not args.input.exists():
            logger.error(f"Directory not found: {args.input}")
            sys.exit(1)
        email_files = list(args.input.glob('*.eml'))

        if not email_files:
            logger.error(f"No .eml files found in {args.input}")
            sys.exit(1)

    logger.info(f"Found {len(email_files)} email file(s) to process")

    # Process emails
    results = []
    for email_file in email_files:
        result = process_email_file(
            email_file,
            classifier,
            args.output,
            verbose=args.verbose
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Exit with error code if any processing failed
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        logger.warning(f"{failed_count} email(s) failed to process")
        sys.exit(1)

    logger.info("Processing complete!")


if __name__ == '__main__':
    main()
