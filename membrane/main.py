"""
FastAPI Application — Trust Layer for LLMs (v3)
================================================
Enterprise-grade AI privacy middleware.

Endpoints:
  POST /secure-llm-call  — Full anonymization pipeline
  GET  /logs             — Audit trail
  GET  /health           — Health check
  GET  /config           — Current config (debug)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Query
from pydantic import BaseModel

from membrane.config import get_config
from membrane.pii.detector import detect_pii
from membrane.tokenizer.tokenizer import tokenize
from membrane.llm.integrity import send_to_llm_with_retry
from membrane.rehydration.rehydrator import rehydrate
from membrane.entity_alignment.alignment import align_entities
from membrane.audit import log_request, get_logs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("trust_layer")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logger.info("🚀 Trust Layer v3 — starting up (presidio=%s, strict=%s, retries=%d)",
                config.enable_presidio, config.strict_mode, config.max_retries)
    yield
    logger.info("🛑 Trust Layer v3 — shutting down")


app = FastAPI(
    title="Trust Layer for LLMs",
    description="Enterprise AI privacy middleware — reversible PII anonymization",
    version="3.1.1",
    lifespan=lifespan,
)

# CORS — allows the Playground UI (separate project) to call this API
from starlette.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SecureCallRequest(BaseModel):
    prompt: str


class IntegrityMetrics(BaseModel):
    placeholders_total: int
    placeholders_preserved: int
    integrity_score: float
    retry_count: int
    status: str


class SecureCallResponse(BaseModel):
    original_prompt: str
    anonymized_prompt: str
    llm_response: str
    final_response: str
    mapping: dict[str, dict[str, Any]]
    metrics: IntegrityMetrics
    alignment: dict | None = None
    error: dict | None = None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s — %.1f ms", request.method, request.url.path, elapsed)
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.1.6"}


@app.get("/config")
async def show_config():
    """Return current config for debugging."""
    config = get_config()
    return {
        "enable_presidio": config.enable_presidio,
        "min_confidence": config.min_confidence,
        "strict_mode": config.strict_mode,
        "max_retries": config.max_retries,
        "logging_enabled": config.logging_enabled,
        "context_window_size": config.context_window_size,
    }


@app.post("/secure-llm-call", response_model=SecureCallResponse)
async def secure_llm_call(body: SecureCallRequest):
    """
    Full pipeline:
    1. Detect PII (Presidio or regex)
    2. Tokenize with entity tracking
    3. LLM call with integrity check + retry
    4. Entity alignment (restore identity from paraphrases)
    5. Rehydration (exact + paraphrase + context)
    6. Audit logging
    """
    original = body.prompt
    logger.info("Received prompt (%d chars)", len(original))

    # Step 1+2: Detect & tokenize
    entities = detect_pii(original)
    logger.info("Detected %d PII entities", len(entities))

    result = tokenize(original, entities)
    logger.info("Anonymized prompt: %s", result.anonymized_text)

    # Step 3: LLM call with integrity checking
    llm_result = send_to_llm_with_retry(result.anonymized_text, result.mapping)
    logger.info(
        "LLM response (score=%.2f, retries=%d, status=%s)",
        llm_result.integrity.score, llm_result.retry_count, llm_result.status,
    )

    # Step 4: Entity alignment (restore identity from paraphrases)
    alignment = align_entities(llm_result.response, result.mapping)
    aligned_text = alignment.text
    if alignment.aligned:
        logger.info(
            "Entity alignment: %d replacements (confidence=%.2f)",
            len(alignment.replacements), alignment.confidence,
        )

    # Step 5: Rehydration
    final, _actions = rehydrate(
        aligned_text, result.mapping, tracker=result.tracker,
    )

    # Step 5: Audit log
    metrics = IntegrityMetrics(
        placeholders_total=llm_result.integrity.total,
        placeholders_preserved=llm_result.integrity.preserved,
        integrity_score=llm_result.integrity.score,
        retry_count=llm_result.retry_count,
        status=llm_result.status,
    )

    log_request(
        original_prompt=original,
        anonymized_prompt=result.anonymized_text,
        llm_response=llm_result.response,
        final_response=final,
        mapping=result.mapping,
        integrity_score=llm_result.integrity.score,
        retry_count=llm_result.retry_count,
        status=llm_result.status,
        entity_tracking=result.tracker.to_dict(),
    )

    # Build alignment metadata
    alignment_meta = None
    if alignment.aligned:
        alignment_meta = {
            "aligned": True,
            "confidence": alignment.confidence,
            "replacements": [
                {"alias": r.alias, "original_value": r.original_value,
                 "entity_type": r.entity_type, "placeholder_key": r.placeholder_key}
                for r in alignment.replacements
            ],
        }

    return SecureCallResponse(
        original_prompt=original,
        anonymized_prompt=result.anonymized_text,
        llm_response=llm_result.response,
        final_response=final,
        mapping=result.mapping,
        metrics=metrics,
        alignment=alignment_meta,
        error=llm_result.error,
    )


@app.get("/logs")
async def logs(limit: int = Query(default=50, ge=1, le=500)):
    """Retrieve audit log entries, most recent first."""
    entries = get_logs(limit=limit)
    return {"count": len(entries), "entries": entries}
