"""
Membrane — Trust Layer SDK
============================
Enterprise AI privacy middleware as an embeddable Python library.
"""

from __future__ import annotations

import os
from typing import Any

from membrane.pii.detector import detect_pii
from membrane.tokenizer.tokenizer import tokenize
from membrane.llm.integrity import send_to_llm_with_retry
from membrane.entity_alignment.alignment import align_entities
from membrane.rehydration.rehydrator import rehydrate
from membrane.audit import log_request
from membrane.config import get_config


class TrustLayer:
    """
    Main interface for the Membrane Python SDK.
    
    Usage:
        layer = TrustLayer(provider="openai", api_key="sk-...")
        response = layer.call("My name is John Doe.")
    """

    def __init__(self, provider: str = "mock", model: str | None = None, api_key: str | None = None, api_base: str | None = None):
        """
        Initialize the TrustLayer.
        
        Parameters
        ----------
        provider : str
            The LLM provider to route calls to ("openai", "anthropic", "gemini", "ollama", "mock").
        model : str | None
            Specific model name override (e.g. "gpt-4", "claude-3-opus", "gemini-1.5-pro").
        api_key : str | None
            The API key for the chosen provider. Can also be set via env vars.
        api_base : str | None
            Base URL override (useful for Ollama or OpenAI-compatible backends).
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        
        # We temporarily inject these into the process environment
        # since the underlying config model is singleton/env-based.
        os.environ["TRUST_LLM_PROVIDER"] = provider
        if model:
            os.environ["TRUST_LLM_MODEL"] = model
        if api_base:
            os.environ["TRUST_LLM_API_BASE"] = api_base

    def call(self, prompt: str) -> dict[str, Any]:
        """
        Run a prompt through the full Trust Layer pipeline and return the result.
        """
        # (1 & 2) Detect and tokenize
        entities = detect_pii(prompt)
        result = tokenize(prompt, entities)
        
        # (3) Call LLM
        if self.api_key:
            _env_key = f"{self.provider.upper()}_API_KEY"
            old_key = os.environ.get(_env_key)
            os.environ[_env_key] = self.api_key

        try:
            llm_result = send_to_llm_with_retry(result.anonymized_text, result.mapping, model=self.model)
        finally:
            if self.api_key:
                if old_key is not None:
                    os.environ[_env_key] = old_key
                else:
                    del os.environ[_env_key]

        # (4) Entity Alignment
        alignment = align_entities(llm_result.response, result.mapping)
        
        # (5) Rehydration
        final, _ = rehydrate(alignment.text, result.mapping, tracker=result.tracker)
        
        # (6) Audit
        log_request(
            original_prompt=prompt,
            anonymized_prompt=result.anonymized_text,
            llm_response=llm_result.response,
            final_response=final,
            mapping=result.mapping,
            integrity_score=llm_result.integrity.score,
            retry_count=llm_result.retry_count,
            status=llm_result.status,
            entity_tracking=result.tracker.to_dict(),
        )

        return {
            "original_prompt": prompt,
            "anonymized_prompt": result.anonymized_text,
            "llm_response": llm_result.response,
            "final_response": final,
            "metrics": {
                "score": llm_result.integrity.score,
                "retries": llm_result.retry_count,
                "status": llm_result.status,
            },
            "alignment": {
                "aligned": alignment.aligned,
                "confidence": alignment.confidence,
            }
        }
