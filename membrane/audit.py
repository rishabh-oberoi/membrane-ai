"""
Audit Logging Module
====================
Records every pipeline request with full context for compliance and debugging.

Storage: In-memory list + JSON-lines file persistence.
Controlled by config.logging_enabled and config.audit_log_file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from membrane.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_log_entries: list[dict[str, Any]] = []
_lock = Lock()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def log_request(
    original_prompt: str,
    anonymized_prompt: str,
    llm_response: str,
    final_response: str,
    mapping: dict[str, dict[str, str]],
    integrity_score: float,
    retry_count: int,
    status: str,
    entity_tracking: dict | None = None,
) -> dict[str, Any]:
    """
    Record a pipeline request in the audit log.

    Returns the log entry dict (or empty dict if logging disabled).
    """
    config = get_config()
    if not config.logging_enabled:
        return {}

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_prompt": original_prompt,
        "anonymized_prompt": anonymized_prompt,
        "llm_response": llm_response,
        "final_response": final_response,
        "mapping": mapping,
        "metrics": {
            "integrity_score": integrity_score,
            "retry_count": retry_count,
            "status": status,
        },
    }
    if entity_tracking:
        entry["entity_tracking"] = entity_tracking

    with _lock:
        _log_entries.append(entry)

    # Persist to file
    audit_file = config.audit_log_file
    if audit_file:
        try:
            with open(audit_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit log to %s: %s", audit_file, exc)

    logger.info(
        "Audit log entry recorded (total: %d, status: %s)",
        len(_log_entries), status,
    )
    return entry


def get_logs(limit: int | None = None) -> list[dict[str, Any]]:
    """Return audit log entries, most recent first."""
    with _lock:
        entries = list(reversed(_log_entries))
    if limit is not None:
        entries = entries[:limit]
    return entries


def clear_logs() -> int:
    """Clear all in-memory log entries. Returns count cleared."""
    with _lock:
        count = len(_log_entries)
        _log_entries.clear()
    logger.info("Cleared %d audit log entries", count)
    return count
