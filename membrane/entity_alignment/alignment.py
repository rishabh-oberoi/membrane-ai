"""
Entity Alignment Layer
======================
Restores identity even when the LLM paraphrases away placeholders.

Problem:
  Input:   "<PERSON_1> emailed the clinic"
  LLM out: "The patient contacted the clinic"   ← placeholder LOST

Solution:
  Detect that "patient" is an alias for PERSON → replace with "John Doe"
  Final:   "John Doe contacted the clinic"

This module sits BETWEEN the LLM response and rehydration in the pipeline.
All logic is deterministic — no LLM involvement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Alias dictionary — maps generic references back to entity types
# ═══════════════════════════════════════════════════════════════

PERSON_ALIASES: list[str] = [
    # Generic
    "person", "individual", "user",
    # Healthcare
    "patient", "diagnosed individual", "referred patient",
    # Business / CRM
    "customer", "client", "employee", "applicant",
    "candidate", "member", "account holder",
    "contact", "sender", "requester",
    # Pronouns
    "he", "she", "they",
]

EMAIL_ALIASES: list[str] = [
    "email", "email address", "e-mail",
    "contact email", "registered email",
    "address",
]

PHONE_ALIASES: list[str] = [
    "phone number", "number", "phone",
    "contact number", "mobile number",
    "direct line", "line",
]

SSN_ALIASES: list[str] = [
    "social security number", "social security", "ssn",
    "social", "tax id", "tax identification number",
]

CREDIT_CARD_ALIASES: list[str] = [
    "credit card number", "credit card", "card number",
    "card", "payment card", "debit card",
    "card on file", "card ending in",
]

LOCATION_ALIASES: list[str] = [
    "location", "address", "city", "place",
    "hometown", "residence", "area",
]

DATE_OF_BIRTH_ALIASES: list[str] = [
    "date of birth", "birth date", "birthday",
    "dob", "born on", "born",
]

URL_ALIASES: list[str] = [
    "website", "url", "link", "web address",
    "site", "webpage", "portal",
]

_ALIAS_MAP: dict[str, list[str]] = {
    "PERSON": PERSON_ALIASES,
    "EMAIL": EMAIL_ALIASES,
    "PHONE": PHONE_ALIASES,
    "SSN": SSN_ALIASES,
    "CREDIT_CARD": CREDIT_CARD_ALIASES,
    "LOCATION": LOCATION_ALIASES,
    "DATE_OF_BIRTH": DATE_OF_BIRTH_ALIASES,
    "URL": URL_ALIASES,
}

# Pronoun aliases need special handling — only replace if they appear
# as a standalone word at the start of a clause or sentence
_PRONOUN_SET = {"he", "she", "they"}


# ═══════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════

@dataclass
class AlignmentReplacement:
    """A single alias→value replacement."""
    alias: str              # The alias word/phrase found
    original_value: str     # The PII value it was replaced with
    entity_type: str        # PERSON, EMAIL, PHONE
    placeholder_key: str    # e.g. "PERSON_1"


@dataclass
class AlignmentResult:
    """Result of entity alignment."""
    text: str                                   # The aligned text
    aligned: bool                               # Whether any alignment was done
    confidence: float                           # Overall confidence (0.0–1.0)
    replacements: list[AlignmentReplacement]    # What was replaced


# ═══════════════════════════════════════════════════════════════
# Context matching
# ═══════════════════════════════════════════════════════════════

def _context_overlap_score(
    llm_output: str,
    context_words: list[str],
) -> float:
    """
    How much of the entity's original context appears in the LLM output?
    Returns 0.0–1.0.
    """
    if not context_words:
        return 0.0
    output_lower = llm_output.lower()
    output_words = set(re.findall(r"[a-z0-9']+", output_lower))
    matches = sum(1 for w in context_words if w in output_words)
    return matches / len(context_words)


# ═══════════════════════════════════════════════════════════════
# Core alignment
# ═══════════════════════════════════════════════════════════════

def align_entities(
    llm_output: str,
    mapping: dict[str, dict],
) -> AlignmentResult:
    """
    Align entities in *llm_output* when placeholders are missing.

    Steps:
    1. Check if placeholders already exist → skip alignment
    2. For each entity type, find alias words in the LLM output
    3. If EXACTLY ONE alias matches for a single-entity type → replace
    4. Score confidence based on match quality and context overlap

    Parameters
    ----------
    llm_output : str
        The raw LLM response text.
    mapping : dict
        PII mapping with optional context:
        {"PERSON_1": {"value": "John Doe", "type": "PERSON", "context": [...]}}

    Returns
    -------
    AlignmentResult
        The aligned text, confidence score, and replacement details.
    """
    # If no mapping, nothing to align
    if not mapping:
        return AlignmentResult(text=llm_output, aligned=False, confidence=1.0, replacements=[])

    # ── Step 1: Check if placeholders are already present ──────────
    all_present = all(f"<{key}>" in llm_output for key in mapping)
    if all_present:
        logger.debug("All placeholders present — skipping alignment")
        return AlignmentResult(text=llm_output, aligned=False, confidence=1.0, replacements=[])

    # ── Step 2: Group entities by type ──────────────────────────────
    entities_by_type: dict[str, list[tuple[str, dict]]] = {}
    for key, info in mapping.items():
        placeholder = f"<{key}>"
        # Only try to align entities whose placeholders are MISSING
        if placeholder not in llm_output:
            etype = info["type"]
            entities_by_type.setdefault(etype, []).append((key, info))

    # ── Step 3: Find and replace aliases ───────────────────────────
    result = llm_output
    replacements: list[AlignmentReplacement] = []
    total_confidence = 0.0
    alignment_count = 0

    for etype, entities in entities_by_type.items():
        # Only align when there's exactly ONE entity of this type missing
        # (otherwise it's ambiguous which value to use)
        if len(entities) != 1:
            logger.debug(
                "Skipping %s alignment: %d entities (ambiguous)",
                etype, len(entities),
            )
            continue

        key, info = entities[0]
        original_value = info["value"]
        context_words = info.get("context", [])
        aliases = _ALIAS_MAP.get(etype, [])

        if not aliases:
            continue

        # Find which aliases appear in the output
        found_aliases: list[tuple[str, int, int]] = []  # (alias, start, end)

        for alias in aliases:
            if alias in _PRONOUN_SET:
                # Pronouns: only match at word boundary, not inside other words
                pattern = re.compile(
                    r"(?<![a-zA-Z])" + re.escape(alias) + r"(?![a-zA-Z])",
                    re.IGNORECASE,
                )
            else:
                # Regular aliases: match as whole phrases
                pattern = re.compile(
                    r"\b" + re.escape(alias) + r"\b",
                    re.IGNORECASE,
                )

            for match in pattern.finditer(result):
                found_aliases.append((alias, match.start(), match.end()))

        if not found_aliases:
            logger.debug("No aliases found for %s in LLM output", key)
            continue

        # Deduplicate: if "the patient" and "patient" both match at overlapping
        # positions, prefer the longer match
        found_aliases.sort(key=lambda x: -(x[2] - x[1]))  # longest first
        used_positions: set[int] = set()
        unique_aliases: list[tuple[str, int, int]] = []

        for alias, start, end in found_aliases:
            if not any(pos in used_positions for pos in range(start, end)):
                unique_aliases.append((alias, start, end))
                used_positions.update(range(start, end))

        # Calculate confidence based on match characteristics
        ctx_score = _context_overlap_score(result, context_words)

        if len(unique_aliases) == 1:
            # Single match — high confidence
            confidence = 0.7 + (ctx_score * 0.2)  # 0.7–0.9
        elif len(unique_aliases) <= 3:
            # A few matches — medium confidence, replace just the first
            confidence = 0.5 + (ctx_score * 0.15)
        else:
            # Too many matches — low confidence, skip
            logger.debug(
                "Too many alias matches for %s (%d) — skipping",
                key, len(unique_aliases),
            )
            continue

        # Replace the FIRST alias occurrence with the original value
        alias_text, start, end = unique_aliases[0]
        # Capitalize if the alias was at the start of a sentence
        replacement_value = original_value
        if start == 0 or result[start - 2:start] == ". ":
            # Preserve capitalization at sentence start
            pass

        result = result[:start] + replacement_value + result[end:]

        replacements.append(AlignmentReplacement(
            alias=alias_text,
            original_value=original_value,
            entity_type=etype,
            placeholder_key=key,
        ))

        total_confidence += confidence
        alignment_count += 1

        logger.info(
            "Aligned '%s' → '%s' (key=%s, confidence=%.2f, context_score=%.2f)",
            alias_text, original_value, key, confidence, ctx_score,
        )

    # ── Step 4: Compute overall confidence ─────────────────────────
    if alignment_count > 0:
        overall_confidence = round(total_confidence / alignment_count, 2)
    else:
        overall_confidence = 0.0

    return AlignmentResult(
        text=result,
        aligned=len(replacements) > 0,
        confidence=overall_confidence,
        replacements=replacements,
    )
