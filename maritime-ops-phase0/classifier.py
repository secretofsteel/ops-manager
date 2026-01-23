"""
Email classification using hybrid rule-based and LLM approach.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Tuple

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
            ValueError: If classification fails completely
        """
        # Stage 1: Try rule-based classification
        logger.debug(f"Attempting rule-based classification for: {email.subject}")
        rule_result, matched_keywords = self._apply_rules(email)

        confidence_threshold = self.rules_config['confidence_threshold']
        if rule_result.confidence >= confidence_threshold:
            logger.info(
                f"Rule-based classification succeeded: {rule_result.category}/{rule_result.subcategory} "
                f"(confidence: {rule_result.confidence:.2f}, keywords: {matched_keywords})"
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
            # Fallback to rule result even with low confidence
            logger.warning("Falling back to rule-based result despite low confidence")
            return rule_result

    def _apply_rules(self, email: ParsedEmail) -> Tuple[ClassificationResult, List[str]]:
        """
        Apply rule-based classification using keywords.

        Args:
            email: Parsed email

        Returns:
            Tuple of (ClassificationResult with source=RULES, list of matched keywords)
        """
        # Combine subject and body for keyword matching
        search_text = f"{email.subject} {email.body_text or ''}"
        search_text_lower = search_text.lower()

        use_word_boundaries = self.rules_config['keyword_matching'].get('use_word_boundaries', True)

        # Find matching keywords
        matches = {}  # category_subcategory -> list of matched keywords
        for keyword, classifications in self.keyword_index.items():
            # Check if keyword matches
            if use_word_boundaries:
                # Use word boundary matching to avoid partial matches
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, search_text_lower):
                    matched = True
                else:
                    matched = False
            else:
                # Simple substring match
                matched = keyword in search_text_lower

            if matched:
                for cls in classifications:
                    key = f"{cls['category']}_{cls['subcategory']}"
                    if key not in matches:
                        matches[key] = []
                    matches[key].append(keyword)

        # Determine best match
        if matches:
            # Sort by number of keyword matches (descending)
            best_match_key = max(matches, key=lambda k: len(matches[k]))
            matched_keywords = matches[best_match_key]
            match_count = len(matched_keywords)

            category, subcategory = best_match_key.split('_', 1)

            # Calculate confidence
            base_conf = self.rules_config['keyword_matching']['base_confidence']
            boost = self.rules_config['keyword_matching']['multi_keyword_boost']
            confidence = min(1.0, base_conf + (match_count - 1) * boost)

            # Try to extract vessel name from subject using common patterns
            vessel_name = self._extract_vessel_name(email.subject)
            
            # Try to extract port from subject/body
            port = self._extract_port_hint(search_text)

            # Detect urgency from keywords
            urgency = self._detect_urgency(search_text_lower)

            return ClassificationResult(
                vessel_name=vessel_name,
                category=category,
                subcategory=subcategory,
                port=port,
                urgency=urgency,
                summary=f"Matched keywords: {', '.join(matched_keywords)}. Review for details.",
                action_required=True,
                confidence=confidence,
                source=ClassificationSource.RULES
            ), matched_keywords

        # No matches - return UNCATEGORIZED with low confidence
        return ClassificationResult(
            category='UNCATEGORIZED',
            urgency=UrgencyLevel.LOW,
            summary="No classification keywords found. Manual review required.",
            action_required=False,
            confidence=0.3,
            source=ClassificationSource.RULES
        ), []

    def _extract_vessel_name(self, text: str) -> Optional[str]:
        """
        Try to extract vessel name from text using common patterns.
        
        Args:
            text: Text to search (usually subject line)
            
        Returns:
            Vessel name if found, None otherwise
        """
        # Common vessel name patterns
        patterns = [
            r'\b(M[/]?V\s+[A-Z][A-Z\s\-\.]+)',  # MV or M/V followed by name
            r'\b(M[/]?T\s+[A-Z][A-Z\s\-\.]+)',  # MT or M/T followed by name
            r'\b(VESSEL\s+[A-Z][A-Z\s\-\.]+)',  # VESSEL followed by name
        ]
        
        text_upper = text.upper()
        
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                vessel_name = match.group(1).strip()
                # Clean up extra spaces
                vessel_name = ' '.join(vessel_name.split())
                # Limit length to avoid grabbing too much
                if len(vessel_name) <= 50:
                    return vessel_name
        
        return None

    def _extract_port_hint(self, text: str) -> Optional[str]:
        """
        Try to extract port name from text.
        
        This is a simple heuristic - LLM will do better.
        
        Args:
            text: Text to search
            
        Returns:
            Port name if found with high confidence, None otherwise
        """
        # Look for common port indicators
        patterns = [
            r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+port\b',  # "at Singapore port"
            r'\bport\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',  # "port of Rotterdam"
            r'\barriv(?:al|ing)\s+(?:at\s+)?([A-Z][a-z]+)\b',     # "arriving at Piraeus"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _detect_urgency(self, text_lower: str) -> UrgencyLevel:
        """
        Detect urgency level from text.
        
        Args:
            text_lower: Lowercased text to analyze
            
        Returns:
            UrgencyLevel
        """
        urgent_keywords = ['urgent', 'urgently', 'asap', 'immediately', 'emergency', 'critical']
        high_keywords = ['as soon as possible', 'time sensitive', 'priority', 'earliest']
        
        for kw in urgent_keywords:
            if kw in text_lower:
                return UrgencyLevel.URGENT
        
        for kw in high_keywords:
            if kw in text_lower:
                return UrgencyLevel.HIGH
        
        return UrgencyLevel.MEDIUM

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
            raise ValueError("LLM client not initialized. Check GOOGLE_API_KEY.")

        # Build categories list for prompt
        categories_list = self._format_categories_for_prompt()

        # Truncate body if too long (keep first 3000 chars)
        body_text = email.body_text or email.body_html or "(empty)"
        if len(body_text) > 3000:
            body_text = body_text[:3000] + "\n...(truncated)"

        # Format the prompt
        prompt = self.prompts['classification_template'].format(
            subject=email.subject,
            sender=email.sender,
            body=body_text,
            categories_list=categories_list
        )

        # Add system instruction
        full_prompt = f"{self.prompts['system_instruction']}\n\n{prompt}"

        try:
            # Configure safety settings to be less restrictive for business emails
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

            # Check if we got a valid response
            if not response.text:
                raise ValueError("Empty response from LLM")

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
        Extract JSON from LLM response, handling various formats.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Clean up the response
        text = response_text.strip()
        
        # Try to find JSON in markdown code block first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object
            # Look for outermost braces
            start_idx = text.find('{')
            if start_idx == -1:
                raise ValueError("No JSON object found in LLM response")
            
            # Find matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(text[start_idx:], start=start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if brace_count != 0:
                raise ValueError("Unbalanced braces in JSON response")
            
            json_str = text[start_idx:end_idx + 1]

        # Clean up common issues
        json_str = json_str.strip()
        
        # Try to parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            # Replace single quotes with double quotes (risky but sometimes works)
            try:
                fixed = json_str.replace("'", '"')
                return json.loads(fixed)
            except:
                pass
            
            logger.error(f"Failed to parse JSON: {json_str[:500]}")
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
            # Validate and set defaults for required fields
            category = result_json.get('category', 'UNCATEGORIZED')
            if not category:
                category = 'UNCATEGORIZED'

            subcategory = result_json.get('subcategory')
            if subcategory == 'null' or subcategory == '':
                subcategory = None

            urgency_str = result_json.get('urgency', 'MEDIUM')
            if not urgency_str:
                urgency_str = 'MEDIUM'
            
            # Parse urgency
            try:
                urgency = UrgencyLevel(urgency_str.upper())
            except ValueError:
                logger.warning(f"Invalid urgency '{urgency_str}', defaulting to MEDIUM")
                urgency = UrgencyLevel.MEDIUM

            summary = result_json.get('summary', 'No summary provided')
            if not summary:
                summary = 'No summary provided'

            action_required = result_json.get('action_required', True)
            if isinstance(action_required, str):
                action_required = action_required.lower() == 'true'

            confidence = result_json.get('confidence', 0.8)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except:
                    confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))

            # Handle dates_mentioned
            dates = result_json.get('dates_mentioned', [])
            if dates is None:
                dates = []
            if isinstance(dates, str):
                dates = [dates] if dates else []

            # Handle vessel_name
            vessel_name = result_json.get('vessel_name')
            if vessel_name in ['null', '', 'None', 'N/A']:
                vessel_name = None

            # Handle port
            port = result_json.get('port')
            if port in ['null', '', 'None', 'N/A']:
                port = None

            return ClassificationResult(
                vessel_name=vessel_name,
                category=category,
                subcategory=subcategory,
                port=port,
                dates_mentioned=dates,
                urgency=urgency,
                summary=summary,
                action_required=action_required,
                confidence=confidence,
                source=ClassificationSource.LLM
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {result_json}")
            raise ValueError(f"Invalid LLM response format: {e}")