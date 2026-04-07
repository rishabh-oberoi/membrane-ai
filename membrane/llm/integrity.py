"""
Integrity Check & Retry Layer
==============================
Validates that all placeholders survive the LLM round-trip.

Flow:
  1. Call LLM
  2. Check which placeholders are present in the response
  3. If any are missing → retry with stronger system prompt
  4. Return structured result with integrity metrics and error details
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from membrane.config import get_config
from membrane.llm.proxy import send_to_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IntegrityResult:
    """Metrics about placeholder preservation."""
    total: int
    preserved: int
    missing: list[str]
    score: float  # preserved / total (1.0 = perfect)


@dataclass
class LLMResult:
    """Full result from an integrity-checked LLM call."""
    response: str
    integrity: IntegrityResult
    retry_count: int
    status: str                              # "ok" | "degraded" | "failed"
    error: dict | None = None                # Structured error when status="failed"


# ---------------------------------------------------------------------------
# Placeholder checking
# ---------------------------------------------------------------------------

def check_placeholders(response: str, mapping: dict[str, dict[str, str]]) -> IntegrityResult:
    """Check which placeholders from *mapping* survived in *response*."""
    total = len(mapping)
    if total == 0:
        return IntegrityResult(total=0, preserved=0, missing=[], score=1.0)

    missing = [k for k in mapping if f"<{k}>" not in response]
    preserved = total - len(missing)
    score = preserved / total if total > 0 else 1.0

    return IntegrityResult(
        total=total, preserved=preserved,
        missing=missing, score=round(score, 2),
    )


# ---------------------------------------------------------------------------
# LLM call with integrity checking and retry
# ---------------------------------------------------------------------------

def send_to_llm_with_retry(
    prompt: str, mapping: dict[str, Any], max_retries: int | None = None, model: str | None = None
) -> LLMResult:
    """
    Send *prompt* to the LLM with automatic retry if placeholders are lost.

    Parameters
    ----------
    prompt : str
        The anonymized prompt.
    mapping : dict
        PII mapping from tokenization.
    max_retries : int or None
        Override for config's max_retries.
    model : str or None
        Override for the LLM model.

    Returns
    -------
    LLMResult
        Response + integrity metrics + structured error if failed.
    """
    config = get_config()
    retries = max_retries if max_retries is not None else config.max_retries

    # First attempt
    response = send_to_llm(prompt, strong_prompt=False, model=model)
    integrity = check_placeholders(response, mapping)
    retry_count = 0

    if integrity.score == 1.0:
        logger.info("All %d placeholders preserved on first attempt", integrity.total)
        return LLMResult(
            response=response, integrity=integrity,
            retry_count=0, status="ok",
        )

    # Retry loop with stronger instructions
    logger.warning(
        "Missing %d/%d placeholders: %s — retrying",
        len(integrity.missing), integrity.total, integrity.missing,
    )

    for attempt in range(1, retries + 1):
        response = send_to_llm(prompt, strong_prompt=True)
        integrity = check_placeholders(response, mapping)
        retry_count = attempt

        if integrity.score == 1.0:
            logger.info("All placeholders preserved after %d retry(ies)", attempt)
            return LLMResult(
                response=response, integrity=integrity,
                retry_count=retry_count, status="ok",
            )

    # Exhausted retries — return structured error
    if integrity.score > 0:
        status = "degraded"
        logger.warning(
            "Partial loss after %d retries (score=%.2f): %s",
            retry_count, integrity.score, integrity.missing,
        )
    else:
        status = "failed"
        logger.error("All placeholders lost after %d retries", retry_count)

    return LLMResult(
        response=response,
        integrity=integrity,
        retry_count=retry_count,
        status=status,
        error={
            "code": "PLACEHOLDER_LOSS",
            "message": f"{len(integrity.missing)}/{integrity.total} placeholders lost after {retry_count} retries",
            "missing_tokens": [f"<{k}>" for k in integrity.missing],
        },
    )
