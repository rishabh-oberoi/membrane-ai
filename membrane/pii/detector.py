"""
PII Detection Module
====================
Production-grade PII detection with Presidio + regex fallback.

Supported entity types:
  PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, LOCATION, DATE_OF_BIRTH, URL

Primary: Microsoft Presidio (spaCy NER + pattern recognizers)
Fallback: Regex-based detection if Presidio is unavailable

Architecture: Exposes a simple `detect_pii()` interface — callers never
need to know which backend is active.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from membrane.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PIIEntity:
    """A single detected PII span."""
    value: str              # The raw text matched
    entity_type: str        # Canonical type: PERSON, EMAIL, PHONE, SSN, etc.
    start: int              # Character offset (inclusive)
    end: int                # Character offset (exclusive)
    confidence: float       # Detection confidence score (0.0 – 1.0)


# ═══════════════════════════════════════════════════════════════
# BACKEND 1: Presidio
# ═══════════════════════════════════════════════════════════════

_analyzer = None
_presidio_available: bool | None = None

_TYPE_MAP: dict[str, str] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE_OF_BIRTH",
    "URL": "URL",
}
_SUPPORTED_ENTITIES = list(_TYPE_MAP.keys())


def _init_presidio() -> bool:
    """Try to initialize Presidio. Returns True if available."""
    global _analyzer, _presidio_available
    if _presidio_available is not None:
        return _presidio_available
    try:
        from presidio_analyzer import AnalyzerEngine
        logger.info("Initializing Presidio AnalyzerEngine...")
        _analyzer = AnalyzerEngine()
        _presidio_available = True
        logger.info("Presidio AnalyzerEngine ready")
    except Exception as exc:
        _presidio_available = False
        logger.warning("Presidio unavailable, falling back to regex: %s", exc)
    return _presidio_available


def _presidio_detect(text: str, min_confidence: float) -> list[PIIEntity]:
    """Detect PII using Presidio."""
    results = _analyzer.analyze(
        text=text, entities=_SUPPORTED_ENTITIES, language="en",
    )
    entities: list[PIIEntity] = []
    for r in results:
        if r.score < min_confidence:
            continue
        canonical = _TYPE_MAP.get(r.entity_type)
        if canonical is None:
            continue
        entities.append(PIIEntity(
            value=text[r.start:r.end],
            entity_type=canonical,
            start=r.start, end=r.end,
            confidence=round(r.score, 2),
        ))
    return entities


# ═══════════════════════════════════════════════════════════════
# BACKEND 2: Regex fallback
# ═══════════════════════════════════════════════════════════════

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[\s\-.]?)?(?:\(\d{3}\)|\d{3})[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)"
)
_NAME_RE = re.compile(r"\b(?:[A-Z][a-z]{1,20})(?:\s+[A-Z][a-z]{1,20}){0,3}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")
_DOB_RE = re.compile(
    r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)

_NAME_STOPWORDS: set[str] = {
    "The", "This", "That", "These", "Those", "What", "Where", "When",
    "Which", "While", "With", "Would", "Will", "Was", "Were", "Who",
    "How", "Have", "Has", "Had", "Here", "His", "Her", "Its",
    "Dear", "Hello", "Hey", "Please", "Sure", "Yes", "Thanks",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October",
    "November", "December", "Today", "Tomorrow", "Yesterday",
    "North", "South", "East", "West",
    "Note", "Important", "Warning", "Error", "Info",
    "Also", "However", "Therefore", "Furthermore", "Moreover",
    "Input", "Output", "Result", "Response", "Request",
    "For", "But", "And", "Not", "Now", "Our", "Can",
    "Let", "Get", "Set", "New", "Old", "All", "Any",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven",
    "Eight", "Nine", "Ten",
    "Later", "Earlier", "Before", "After", "Since", "Until",
    "Patient", "Client", "User", "Admin", "Contact",
    "Regards", "Sincerely", "Best", "Based", "Here",
    "Customer", "Type", "Corp",
}


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _regex_detect(text: str) -> list[PIIEntity]:
    """Detect PII using regex patterns (fallback)."""
    entities: list[PIIEntity] = []

    for m in _EMAIL_RE.finditer(text):
        entities.append(PIIEntity(
            value=m.group(), entity_type="EMAIL",
            start=m.start(), end=m.end(), confidence=0.95,
        ))

    for m in _PHONE_RE.finditer(text):
        entities.append(PIIEntity(
            value=m.group(), entity_type="PHONE",
            start=m.start(), end=m.end(), confidence=0.85,
        ))

    for m in _SSN_RE.finditer(text):
        entities.append(PIIEntity(
            value=m.group(), entity_type="SSN",
            start=m.start(), end=m.end(), confidence=0.9,
        ))

    for m in _CC_RE.finditer(text):
        digit_count = sum(c.isdigit() for c in m.group())
        if digit_count in (15, 16):  # Amex=15, Visa/MC=16
            entities.append(PIIEntity(
                value=m.group(), entity_type="CREDIT_CARD",
                start=m.start(), end=m.end(), confidence=0.85,
            ))

    for m in _DOB_RE.finditer(text):
        entities.append(PIIEntity(
            value=m.group(), entity_type="DATE_OF_BIRTH",
            start=m.start(), end=m.end(), confidence=0.7,
        ))

    for m in _URL_RE.finditer(text):
        entities.append(PIIEntity(
            value=m.group(), entity_type="URL",
            start=m.start(), end=m.end(), confidence=0.9,
        ))

    occupied = [(e.start, e.end) for e in entities]
    for m in _NAME_RE.finditer(text):
        candidate = m.group()
        if candidate in _NAME_STOPWORDS:
            continue
        words = candidate.split()
        if len(words) == 1 and len(candidate) <= 3:
            continue
        if any(_spans_overlap(m.start(), m.end(), s, e) for s, e in occupied):
            continue
        entities.append(PIIEntity(
            value=candidate, entity_type="PERSON",
            start=m.start(), end=m.end(), confidence=0.7,
        ))

    return entities


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def detect_pii(text: str) -> list[PIIEntity]:
    """
    Scan *text* for PII entities.

    Runs Presidio if enabled, but ALWAYS runs Regex detection immediately 
    afterwards to catch highly-structured data that Presidio's NLP models 
    might miss (like strictly-formatted US SSNs, Credit Cards, and URLs).
    Returns merged entities sorted by position.
    """
    config = get_config()

    entities: list[PIIEntity] = []
    backend = "regex_only"

    # 1. Run Presidio NLP (Great for Contextual PERSON and LOCATION)
    if config.enable_presidio and _init_presidio():
        entities.extend(_presidio_detect(text, config.min_confidence))
        backend = "presidio+regex"

    # 2. Run highly-optimized Regex Engine (Flawless for SSN, Email, CC)
    regex_entities = _regex_detect(text)
    
    # 3. Deduplicate (prefer Presidio if spans overlap)
    occupied = [(e.start, e.end) for e in entities]
    for re_ent in regex_entities:
        if not any(_spans_overlap(re_ent.start, re_ent.end, s, e) for s, e in occupied):
            entities.append(re_ent)

    # 4. Sort strictly by position
    entities.sort(key=lambda e: e.start)

    logger.info(
        "Detected %d PII entities via %s: %s",
        len(entities), backend,
        [(e.entity_type, e.value, e.confidence) for e in entities],
    )
    return entities
