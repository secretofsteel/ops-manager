"""
Email classification using hybrid rule-based and LLM approach.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from models import ParsedEmail, ClassificationResult, UrgencyLevel, ClassificationSource

logger = logging.getLogger(__name__)


class EmailClassifier:
    """Hybrid email classifier using rules and LLM."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize classifier with configuration.

        Args:
            config: Configuration dictionary loaded from config.yaml
        """
        self.config = config
        self.categories = config['categories']
        self.rules_config = config['rules']
        self.prompts = config['prompts']
        self.llm_settings = config['llm_settings']

        # Build keyword index for rule-based classification
        self._build_keyword_index()

        # Initialize Gemini API
        self._init_llm()

    def _build_keyword_index(self):
        """Build an index of keywords to categories/subcategories."""
        self.keyword_index = {}

        for category, cat_data in self.categories.items():
            if 'subcategories' in cat_data:
                for subcategory, subcat_data in cat_data['subcategories'].items():
                    if 'keywords' in subcat_data:
                        for keyword in subcat_data['keywords']:
                            key = keyword.lower()
                            if key not in self.keyword_index:
                                self.keyword_index[key] = []
                            self.keyword_index[key].append({
                                'category': category,
                                'subcategory': subcategory
                            })

        logger.info(f"Built keyword index with {len(self.keyword_index)} keywords")

    def _init_llm(self):
        """Initialize Gemini API client."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set. LLM classification will fail.")
            self.llm_client = None
            return

        try:
            genai.configure(api_key=api_key)
            self.llm_client = genai.GenerativeModel(
                model_name=self.llm_settings['model_name']
            )
            logger.info(f"Initialized Gemini model: {self.llm_settings['model_name']}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            self.llm_client = None

    def classify(self, email: ParsedEmail) -> ClassificationResult:
        """
        Classify an email using hybrid approach.

        Args:
            email: Parsed email to classify

        Returns:
            ClassificationResult

        Raises:
            ValueError: If classification fails
        """
        # Stage 1: Try rule-based classification
        logger.debug(f"Attempting rule-based classification for: {email.subject}")
        rule_result = self._apply_rules(email)

        confidence_threshold = self.rules_config['confidence_threshold']
        if rule_result.confidence >= confidence_threshold:
            logger.info(
                f"Rule-based classification succeeded with confidence {rule_result.confidence:.2f}"
            )
            return rule_result

        # Stage 2: Use LLM for uncertain cases
        logger.info(
            f"Rule-based confidence {rule_result.confidence:.2f} below threshold "
            f"{confidence_threshold:.2f}, using LLM"
        )

        try:
            llm_result = self._llm_classify(email)
            return llm_result
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to rule result with reduced confidence
            logger.warning("Falling back to rule-based result")
            return rule_result

    def _apply_rules(self, email: ParsedEmail) -> ClassificationResult:
        """
        Apply rule-based classification using keywords.

        Args:
            email: Parsed email

        Returns:
            ClassificationResult with source=RULES
        """
        # Combine subject and body for keyword matching
        search_text = f"{email.subject} {email.body_text or ''}"

        if not self.rules_config['keyword_matching']['case_sensitive']:
            search_text = search_text.lower()

        # Find matching keywords
        matches = {}  # category_subcategory -> count
        for keyword, classifications in self.keyword_index.items():
            if keyword in search_text:
                for cls in classifications:
                    key = f"{cls['category']}_{cls['subcategory']}"
                    matches[key] = matches.get(key, 0) + 1

        # Determine best match
        if matches:
            # Sort by count (descending)
            best_match_key = max(matches, key=matches.get)
            match_count = matches[best_match_key]

            category, subcategory = best_match_key.split('_', 1)

            # Calculate confidence
            base_conf = self.rules_config['keyword_matching']['base_confidence']
            boost = self.rules_config['keyword_matching']['multi_keyword_boost']
            confidence = min(1.0, base_conf + (match_count - 1) * boost)

            return ClassificationResult(
                category=category,
                subcategory=subcategory,
                urgency=UrgencyLevel.MEDIUM,  # Default urgency
                summary=f"Classified as {category}/{subcategory} based on keyword matching",
                action_required=True,  # Assume action needed for matched categories
                confidence=confidence,
                source=ClassificationSource.RULES
            )

        # No matches - return UNCATEGORIZED with low confidence
        return ClassificationResult(
            category='UNCATEGORIZED',
            urgency=UrgencyLevel.LOW,
            summary="No keywords matched, classified as uncategorized",
            action_required=False,
            confidence=0.3,
            source=ClassificationSource.RULES
        )

    def _llm_classify(self, email: ParsedEmail) -> ClassificationResult:
        """
        Classify email using Gemini LLM.

        Args:
            email: Parsed email

        Returns:
            ClassificationResult with source=LLM

        Raises:
            ValueError: If LLM client not initialized or API call fails
        """
        if self.llm_client is None:
            raise ValueError("LLM client not initialized")

        # Build categories list for prompt
        categories_list = self._format_categories_for_prompt()

        # Format the prompt
        prompt = self.prompts['classification_template'].format(
            subject=email.subject,
            sender=email.sender,
            body=email.body_text or email.body_html or "(empty)",
            categories_list=categories_list
        )

        # Add system instruction
        full_prompt = f"{self.prompts['system_instruction']}\n\n{prompt}"

        try:
            # Configure safety settings to be less restrictive
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Generate response
            response = self.llm_client.generate_content(
                full_prompt,
                generation_config={
                    'temperature': self.llm_settings['temperature'],
                    'max_output_tokens': self.llm_settings['max_tokens'],
                },
                safety_settings=safety_settings
            )

            # Extract and parse JSON response
            result_json = self._extract_json_from_response(response.text)
            return self._parse_llm_response(result_json)

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise ValueError(f"LLM classification failed: {e}")

    def _format_categories_for_prompt(self) -> str:
        """Format categories structure for LLM prompt."""
        lines = []
        for category, cat_data in self.categories.items():
            lines.append(f"\n{category}: {cat_data['description']}")
            if 'subcategories' in cat_data and cat_data['subcategories']:
                for subcategory, subcat_data in cat_data['subcategories'].items():
                    lines.append(f"  - {subcategory}: {subcat_data['description']}")

        return "\n".join(lines)

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extract JSON from LLM response, handling markdown code blocks.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in LLM response")

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {json_str}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

    def _parse_llm_response(self, result_json: dict) -> ClassificationResult:
        """
        Parse LLM JSON response into ClassificationResult.

        Args:
            result_json: Dictionary from LLM

        Returns:
            ClassificationResult

        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Validate required fields
            if 'category' not in result_json:
                raise ValueError("Missing required field: category")
            if 'urgency' not in result_json:
                result_json['urgency'] = 'MEDIUM'
            if 'summary' not in result_json:
                raise ValueError("Missing required field: summary")
            if 'action_required' not in result_json:
                result_json['action_required'] = True

            # Parse urgency
            urgency_str = result_json['urgency'].upper()
            try:
                urgency = UrgencyLevel(urgency_str)
            except ValueError:
                logger.warning(f"Invalid urgency '{urgency_str}', defaulting to MEDIUM")
                urgency = UrgencyLevel.MEDIUM

            return ClassificationResult(
                vessel_name=result_json.get('vessel_name'),
                category=result_json['category'],
                subcategory=result_json.get('subcategory'),
                port=result_json.get('port'),
                dates_mentioned=result_json.get('dates_mentioned', []),
                urgency=urgency,
                summary=result_json['summary'],
                action_required=result_json['action_required'],
                confidence=result_json.get('confidence', 0.9),
                source=ClassificationSource.LLM
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"Invalid LLM response format: {e}")
