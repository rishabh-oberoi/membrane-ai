"""
Config System
=============
Centralized configuration for the Trust Layer.

All tunables are loaded from environment variables with sane defaults.
Modules use `get_config()` to access settings — no hardcoded values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """Immutable configuration for the Trust Layer pipeline."""

    # PII Detection
    enable_presidio: bool = True        # Use Presidio; False = regex fallback
    min_confidence: float = 0.4         # Minimum confidence threshold for detections

    # LLM Proxy
    llm_provider: str = "mock"          # Provider: openai, anthropic, gemini, ollama, mock
    llm_api_base: str | None = None     # Base URL (e.g., for Ollama)
    llm_model: str | None = None        # Model name (e.g., gpt-4, claude-3-opus, gemini-pro)
    strict_mode: bool = True            # Prepend preservation instructions to every call
    max_retries: int = 1                # Retries on placeholder loss (0 = no retry)

    # Audit
    logging_enabled: bool = True        # Enable audit log recording
    audit_log_file: str = "audit_log.json"  # Path for JSON-lines audit log ("" = disable)

    # Entity Tracker
    context_window_size: int = 5        # Words of context captured around each entity


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_config: Config | None = None


def get_config() -> Config:
    """
    Return the global Config instance, loading from env vars on first call.

    Env var mapping (all optional):
        TRUST_ENABLE_PRESIDIO   = "true" | "false"
        TRUST_MIN_CONFIDENCE    = float
        TRUST_LLM_PROVIDER      = "openai" | "anthropic" | "gemini" | "ollama" | "mock"
        TRUST_LLM_API_BASE      = str
        TRUST_STRICT_MODE       = "true" | "false"
        TRUST_MAX_RETRIES       = int
        TRUST_LOGGING_ENABLED   = "true" | "false"
        AUDIT_LOG_FILE          = str
        TRUST_CONTEXT_WINDOW    = int
    """
    global _config
    if _config is not None:
        return _config

    def _bool(key: str, default: bool) -> bool:
        val = os.environ.get(key, "").lower()
        if val in ("true", "1", "yes"):
            return True
        if val in ("false", "0", "no"):
            return False
        return default

    _config = Config(
        enable_presidio=_bool("TRUST_ENABLE_PRESIDIO", True),
        min_confidence=float(os.environ.get("TRUST_MIN_CONFIDENCE", "0.4")),
        llm_provider=os.environ.get("TRUST_LLM_PROVIDER", "mock").lower(),
        llm_api_base=os.environ.get("TRUST_LLM_API_BASE"),
        llm_model=os.environ.get("TRUST_LLM_MODEL"),
        strict_mode=_bool("TRUST_STRICT_MODE", True),
        max_retries=int(os.environ.get("TRUST_MAX_RETRIES", "1")),
        logging_enabled=_bool("TRUST_LOGGING_ENABLED", True),
        audit_log_file=os.environ.get("AUDIT_LOG_FILE", "audit_log.json"),
        context_window_size=int(os.environ.get("TRUST_CONTEXT_WINDOW", "5")),
    )
    return _config


def reset_config() -> None:
    """Reset the cached config (useful for tests)."""
    global _config
    _config = None
