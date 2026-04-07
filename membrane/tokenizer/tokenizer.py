"""
Tokenization Module
===================
Replaces detected PII entities with stable, typed placeholders.
Maintains a bidirectional mapping AND populates the EntityTracker
with context windows for downstream rehydration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from membrane.pii.detector import PIIEntity
from membrane.entity_tracker.tracker import EntityTracker


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TokenizationResult:
    """Holds the anonymized text, PII mapping, and entity tracker."""
    anonymized_text: str
    mapping: dict[str, dict[str, str]]  # {"PERSON_1": {"value": "John", "type": "PERSON"}}
    tracker: EntityTracker              # Context-enriched entity registry


# ---------------------------------------------------------------------------
# Core tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str, entities: list[PIIEntity]) -> TokenizationResult:
    """
    Replace each PII entity in *text* with a placeholder like <PERSON_1>.

    - Entities with the **same value** get the **same placeholder** (stable).
    - Builds an EntityTracker with ±N word context windows.
    - Replacements are applied right-to-left so character offsets stay valid.
    """
    type_counters: dict[str, int] = {}
    value_to_placeholder: dict[str, str] = {}
    mapping: dict[str, dict[str, str]] = {}
    tracker = EntityTracker()

    # First pass: assign placeholders and track entities
    entity_placeholders: list[tuple[PIIEntity, str]] = []

    for entity in entities:
        if entity.value in value_to_placeholder:
            placeholder = value_to_placeholder[entity.value]
        else:
            etype = entity.entity_type
            type_counters[etype] = type_counters.get(etype, 0) + 1
            key = f"{etype}_{type_counters[etype]}"
            placeholder = f"<{key}>"
            value_to_placeholder[entity.value] = placeholder
            mapping[key] = {"value": entity.value, "type": etype}

            # Register in entity tracker with context
            tracker.track(
                placeholder_key=key,
                value=entity.value,
                entity_type=etype,
                confidence=entity.confidence,
                start=entity.start,
                end=entity.end,
                original_text=text,
            )

        entity_placeholders.append((entity, placeholder))

    # Second pass: replace right-to-left
    anonymized = text
    for entity, placeholder in reversed(entity_placeholders):
        anonymized = anonymized[:entity.start] + placeholder + anonymized[entity.end:]

    # Enrich mapping with context words from tracker (for entity alignment)
    for key in mapping:
        tracked = tracker.get(key)
        if tracked:
            mapping[key]["context"] = tracked.context_before + tracked.context_after

    return TokenizationResult(
        anonymized_text=anonymized,
        mapping=mapping,
        tracker=tracker,
    )
