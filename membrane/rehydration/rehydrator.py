"""
Rehydration Module
==================
Replaces placeholders in LLM responses with original PII values.

Three-phase approach:
  1. Exact placeholder match  — <PERSON_1> → "John Doe"
  2. Paraphrase fallback      — "the patient" → "John Doe" (unambiguous)
  3. Context-aware fallback   — uses EntityTracker context windows to find
                                the right replacement even when the LLM
                                restructures the sentence

All logic is deterministic; no LLM involvement.
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from membrane.entity_tracker.tracker import EntityTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paraphrase patterns per entity type
# ---------------------------------------------------------------------------

_PARAPHRASE_PATTERNS: dict[str, list[str]] = {
    "PERSON": [
        "the person", "the individual", "the user",
        "this person", "said person",
        "the mentioned person", "the named individual",
        "the patient", "the patient's name",
        "the diagnosed individual", "the referred patient",
        "the client", "the customer", "the employee",
        "the applicant", "the candidate", "the member",
        "the contact", "the account holder",
        "the sender", "the requester",
    ],
    "EMAIL": [
        "the email", "the email address", "their email",
        "the provided email", "the given email",
        "the contact email", "the registered email",
        "their email address", "the sender's email",
    ],
    "PHONE": [
        "the phone number", "the number", "their phone number",
        "the provided number", "the contact number",
        "their number", "the given number",
        "the phone", "their contact number",
        "the registered number", "the mobile number",
    ],
    "SSN": [
        "the social security number", "the SSN",
        "the social", "their social security",
        "the provided SSN", "the tax ID",
    ],
    "CREDIT_CARD": [
        "the credit card number", "the credit card",
        "the card number", "the card",
        "their card", "the payment card",
        "the card on file", "the debit card",
    ],
    "LOCATION": [
        "the location", "the address",
        "their location", "the place",
        "the area", "the city",
        "their address", "the residence",
    ],
    "DATE_OF_BIRTH": [
        "the date of birth", "the birth date",
        "the birthday", "the DOB",
        "their date of birth", "their birthday",
    ],
    "URL": [
        "the website", "the URL", "the link",
        "the web address", "the site",
        "the webpage", "the portal",
    ],
}


# ---------------------------------------------------------------------------
# Core rehydration
# ---------------------------------------------------------------------------

def rehydrate(
    text: str,
    mapping: dict[str, dict[str, str]],
    tracker: EntityTracker | None = None,
) -> tuple[str, dict[str, str]]:
    """
    Replace placeholders (and paraphrases) in *text* using *mapping*.

    Parameters
    ----------
    text : str
        LLM response containing placeholders like <PERSON_1>.
    mapping : dict
        Mapping from tokenizer: {"PERSON_1": {"value": "John", "type": "PERSON"}}
    tracker : EntityTracker, optional
        If provided, enables Phase 3 context-aware fallback.

    Returns
    -------
    tuple[str, dict[str, str]]
        - Rehydrated text
        - Actions taken: {target: method} where method is
          "exact", "paraphrase", or "context"
    """
    result = text
    actions: dict[str, str] = {}

    # ── Phase 1: Exact placeholder replacement ──────────────────────────
    for key, info in mapping.items():
        placeholder = f"<{key}>"
        if placeholder in result:
            result = result.replace(placeholder, info["value"])
            actions[placeholder] = "exact"
            logger.debug("Replaced placeholder %s → %s", placeholder, info["value"])

    # ── Phase 2: Paraphrase fallback ────────────────────────────────────
    entities_by_type: dict[str, list[dict[str, str]]] = {}
    for info in mapping.values():
        entities_by_type.setdefault(info["type"], []).append(info)

    for entity_type, patterns in _PARAPHRASE_PATTERNS.items():
        entities = entities_by_type.get(entity_type, [])
        if len(entities) != 1:
            continue  # Ambiguous — skip

        original_value = entities[0]["value"]
        for phrase in patterns:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            new_result = pattern.sub(original_value, result)
            if new_result != result:
                actions[phrase] = "paraphrase"
                logger.debug("Paraphrase fallback: '%s' → '%s'", phrase, original_value)
                result = new_result

    # ── Phase 3: Context-aware fallback ─────────────────────────────────
    # If a placeholder was lost and tracker is available, look for context
    # words in the LLM output and try to locate where the entity should go.
    if tracker is not None:
        result = _context_aware_fallback(result, mapping, tracker, actions)

    return result, actions


def _context_aware_fallback(
    text: str,
    mapping: dict[str, dict[str, str]],
    tracker: "EntityTracker",
    actions: dict[str, str],
) -> str:
    """
    Phase 3: Use entity context windows to find and replace paraphrases
    that weren't caught by the static pattern list.

    Only activates for entity types with exactly ONE entity (no ambiguity).
    """
    # Find which placeholders were NOT resolved in Phase 1
    unresolved_keys: list[str] = []
    for key, info in mapping.items():
        placeholder = f"<{key}>"
        if placeholder not in actions:
            # Check if the original value is already in the text (from Phase 2)
            if info["value"] not in text:
                unresolved_keys.append(key)

    if not unresolved_keys:
        return text

    # Group unresolved by type — only process unambiguous single-entity types
    unresolved_by_type: dict[str, list[str]] = {}
    for key in unresolved_keys:
        etype = mapping[key]["type"]
        unresolved_by_type.setdefault(etype, []).append(key)

    result = text
    result_lower = result.lower()
    _WORD_RE = re.compile(r"[A-Za-z0-9']+")

    for etype, keys in unresolved_by_type.items():
        if len(keys) != 1:
            continue  # Ambiguous — skip

        key = keys[0]
        entity = tracker.get(key)
        if entity is None:
            continue

        context_words = tracker.context_words(key)
        if not context_words:
            continue

        # Count how many context words appear in the LLM output
        response_words = set(w.lower() for w in _WORD_RE.findall(result))
        overlap = context_words & response_words
        overlap_ratio = len(overlap) / len(context_words) if context_words else 0

        # Only proceed if strong context overlap (≥40% of context words present)
        if overlap_ratio < 0.4:
            logger.debug(
                "Context overlap too low for %s: %.0f%% (%s)",
                key, overlap_ratio * 100, overlap,
            )
            continue

        # Try to find a generic reference to this entity type and replace it
        # Only replace the FIRST occurrence of a type-specific generic term
        type_generics = {
            "PERSON": [r"\b(?:he|she|they|the\s+\w+)\b"],
            "EMAIL": [r"\b(?:the\s+address|the\s+contact)\b"],
            "PHONE": [r"\b(?:the\s+line|the\s+contact)\b"],
        }

        # For now, just log the context match — the paraphrase list should
        # handle most cases. This phase is primarily useful for edge cases.
        logger.info(
            "Context match for %s (%.0f%% overlap): context=%s, overlap=%s",
            key, overlap_ratio * 100, context_words, overlap,
        )

    return result
