"""
Email parsing functionality for .eml files.
"""

import email
import logging
from email import policy
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import re

from models import ParsedEmail, AttachmentInfo

logger = logging.getLogger(__name__)


def parse_eml_file(path: Path) -> ParsedEmail:
    """
    Parse an .eml file and extract structured information.

    Args:
        path: Path to the .eml file

    Returns:
        ParsedEmail object with extracted data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed
    """
    if not path.exists():
        raise FileNotFoundError(f"Email file not found: {path}")

    try:
        # Read and parse the email with modern policy
        with open(path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        # Extract basic headers
        message_id = msg.get('Message-ID')
        subject = msg.get('Subject', '(No Subject)')
        sender = msg.get('From', '(Unknown Sender)')

        # Parse recipients
        recipients = []
        for header in ['To', 'Cc']:
            if msg.get(header):
                recipients.extend(_extract_addresses(msg.get(header)))

        # Parse date
        received_at = None
        if msg.get('Date'):
            try:
                received_at = email.utils.parsedate_to_datetime(msg.get('Date'))
            except Exception as e:
                logger.warning(f"Failed to parse date '{msg.get('Date')}': {e}")

        # Extract body content
        body_text, body_html = _extract_body(msg)

        # Extract attachment metadata
        attachments = _extract_attachments(msg)

        return ParsedEmail(
            message_id=message_id,
            subject=subject,
            sender=sender,
            recipients=recipients,
            received_at=received_at,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            raw_path=path
        )

    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")
        raise ValueError(f"Failed to parse email: {e}")


def _extract_addresses(header_value: str) -> List[str]:
    """Extract email addresses from a header value."""
    addresses = []
    try:
        for name, addr in email.utils.getaddresses([header_value]):
            if addr:
                addresses.append(addr)
    except Exception as e:
        logger.warning(f"Failed to parse addresses from '{header_value}': {e}")
    return addresses


def _extract_body(msg: EmailMessage) -> tuple[Optional[str], Optional[str]]:
    """
    Extract plain text and HTML body from email message.

    Args:
        msg: Email message object

    Returns:
        Tuple of (body_text, body_html)
    """
    body_text = None
    body_html = None

    try:
        # Try to get plain text body first
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))

                # Skip attachments
                if 'attachment' in content_disposition:
                    continue

                if content_type == 'text/plain' and body_text is None:
                    body_text = _decode_payload(part)
                elif content_type == 'text/html' and body_html is None:
                    body_html = _decode_payload(part)
        else:
            # Single part message
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                body_text = _decode_payload(msg)
            elif content_type == 'text/html':
                body_html = _decode_payload(msg)

        # If we only have HTML, try to strip tags for text version
        if body_text is None and body_html is not None:
            body_text = _strip_html_tags(body_html)

    except Exception as e:
        logger.warning(f"Error extracting body: {e}")

    return body_text, body_html


def _decode_payload(part: EmailMessage) -> Optional[str]:
    """
    Decode email part payload handling various encodings.

    Args:
        part: Email message part

    Returns:
        Decoded string or None if decoding fails
    """
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return None

        # Try to decode with specified charset
        charset = part.get_content_charset()
        if charset:
            try:
                return payload.decode(charset, errors='replace')
            except (LookupError, UnicodeDecodeError):
                pass

        # Fallback to common encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return payload.decode(encoding, errors='replace')
            except (UnicodeDecodeError, AttributeError):
                continue

        # Last resort: decode with replacement
        if isinstance(payload, bytes):
            return payload.decode('utf-8', errors='replace')
        return str(payload)

    except Exception as e:
        logger.warning(f"Failed to decode payload: {e}")
        return None


def _strip_html_tags(html: str) -> str:
    """
    Strip HTML tags to get plain text.

    Args:
        html: HTML string

    Returns:
        Plain text with tags removed
    """
    # Remove script and style elements
    clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', clean)

    # Decode HTML entities
    clean = clean.replace('&nbsp;', ' ')
    clean = clean.replace('&lt;', '<')
    clean = clean.replace('&gt;', '>')
    clean = clean.replace('&amp;', '&')
    clean = clean.replace('&quot;', '"')

    # Clean up whitespace
    clean = re.sub(r'\s+', ' ', clean)

    return clean.strip()


def _extract_attachments(msg: EmailMessage) -> List[AttachmentInfo]:
    """
    Extract attachment metadata from email.

    Args:
        msg: Email message object

    Returns:
        List of AttachmentInfo objects
    """
    attachments = []

    try:
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get('Content-Disposition', ''))

                # Check if this is an attachment
                if 'attachment' in content_disposition or part.get_filename():
                    filename = part.get_filename()
                    if filename:
                        content_type = part.get_content_type()

                        # Try to get size
                        size_bytes = 0
                        payload = part.get_payload(decode=True)
                        if payload:
                            size_bytes = len(payload)

                        attachments.append(AttachmentInfo(
                            filename=filename,
                            content_type=content_type,
                            size_bytes=size_bytes
                        ))

    except Exception as e:
        logger.warning(f"Error extracting attachments: {e}")

    return attachments
