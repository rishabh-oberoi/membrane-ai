"""
Entity Tracker Module
=====================
Tracks each PII entity across the pipeline with surrounding context.

The context window (±N words around each entity) is the key differentiator —
it enables context-aware rehydration when the LLM paraphrases placeholders.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from membrane.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrackedEntity:
    """A PII entity enriched with pipeline tracking metadata."""
    placeholder_key: str        # e.g. "PERSON_1"
    value: str                  # Original PII value
    entity_type: str            # "PERSON", "EMAIL", "PHONE"
    confidence: float           # Detection confidence (0.0–1.0)
    start: int                  # Character offset in original text
    end: int                    # Character offset in original text
    context_before: list[str]   # N words before the entity
    context_after: list[str]    # N words after the entity


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

# Splits on whitespace and punctuation boundaries while keeping words clean
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _extract_context(
    text: str,
    start: int,
    end: int,
    window: int,
) -> tuple[list[str], list[str]]:
    """
    Extract up to *window* words before and after the span [start, end] in *text*.

    Returns (words_before, words_after) as lowercased lists.
    """
    before_text = text[:start]
    after_text = text[end:]

    words_before = [w.lower() for w in _WORD_RE.findall(before_text)]
    words_after = [w.lower() for w in _WORD_RE.findall(after_text)]

    return words_before[-window:], words_after[:window]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class EntityTracker:
    """
    Maintains a registry of tracked entities for a single pipeline request.

    Built during tokenization, consumed during rehydration.
    """

    def __init__(self) -> None:
        self._entities: dict[str, TrackedEntity] = {}  # key → TrackedEntity

    def track(
        self,
        placeholder_key: str,
        value: str,
        entity_type: str,
        confidence: float,
        start: int,
        end: int,
        original_text: str,
    ) -> TrackedEntity:
        """
        Register an entity in the tracker.

        Parameters
        ----------
        placeholder_key : str
            e.g. "PERSON_1"
        original_text : str
            The full original text (needed for context extraction).
        """
        config = get_config()
        ctx_before, ctx_after = _extract_context(
            original_text, start, end, config.context_window_size,
        )

        entity = TrackedEntity(
            placeholder_key=placeholder_key,
            value=value,
            entity_type=entity_type,
            confidence=confidence,
            start=start,
            end=end,
            context_before=ctx_before,
            context_after=ctx_after,
        )
        self._entities[placeholder_key] = entity

        logger.debug(
            "Tracked %s: '%s' context=[%s | %s]",
            placeholder_key, value,
            " ".join(ctx_before), " ".join(ctx_after),
        )
        return entity

    def get(self, key: str) -> TrackedEntity | None:
        """Get a tracked entity by placeholder key."""
        return self._entities.get(key)

    def all_entities(self) -> list[TrackedEntity]:
        """Return all tracked entities, ordered by position."""
        return sorted(self._entities.values(), key=lambda e: e.start)

    def by_type(self, entity_type: str) -> list[TrackedEntity]:
        """Return all entities of a given type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def context_words(self, key: str) -> set[str]:
        """Return the combined context words for an entity (for matching)."""
        entity = self._entities.get(key)
        if entity is None:
            return set()
        return set(entity.context_before) | set(entity.context_after)

    def to_dict(self) -> dict[str, dict]:
        """Serialize the tracker for logging / API responses."""
        return {
            key: {
                "value": e.value,
                "type": e.entity_type,
                "confidence": e.confidence,
                "context_before": e.context_before,
                "context_after": e.context_after,
            }
            for key, e in self._entities.items()
        }

    def __len__(self) -> int:
        return len(self._entities)

    def __repr__(self) -> str:
        return f"EntityTracker({len(self._entities)} entities)"
