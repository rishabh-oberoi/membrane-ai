"""
LLM Proxy Module
================
Sends anonymized prompts to an LLM backend with placeholder-preservation
instructions automatically prepended.

Behavior controlled by config:
  - strict_mode: prepend preservation instructions
  - llm_provider: Which API to route to (openai, anthropic, gemini, ollama, mock)

The system prompt is configurable via env vars.
"""

from __future__ import annotations

import logging
import os
import json
import httpx

from membrane.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder preservation prompts
# ---------------------------------------------------------------------------

_DEFAULT_PRESERVATION_PROMPT = (
    "CRITICAL INSTRUCTION: The user message contains placeholder tokens like "
    "<PERSON_1>, <EMAIL_1>, <PHONE_1>, etc. You MUST preserve these tokens "
    "EXACTLY as they appear. Do NOT modify, remove, rephrase, or expand them. "
    "Include them verbatim in your response whenever you reference the "
    "corresponding entity."
)

_STRONG_PRESERVATION_PROMPT = (
    "CRITICAL: Your previous response LOST placeholder tokens. This is "
    "unacceptable. The user message contains tokens like <PERSON_1>, <EMAIL_1>, "
    "<PHONE_1>. You MUST copy these tokens EXACTLY into your response. "
    "Do NOT replace them with real names, addresses, or descriptions. "
    "Do NOT paraphrase them. Reproduce them CHARACTER FOR CHARACTER."
)


def get_preservation_prompt(strong: bool = False) -> str:
    """Return the system-level placeholder preservation instruction."""
    if strong:
        return os.environ.get("PLACEHOLDER_STRONG_PROMPT", _STRONG_PRESERVATION_PROMPT)
    return os.environ.get("PLACEHOLDER_SYSTEM_PROMPT", _DEFAULT_PRESERVATION_PROMPT)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def _mock_response(prompt: str) -> str:
    """Deterministic mock response that preserves placeholders."""
    return (
        f"Thank you for your message. Based on the information provided, "
        f"I can confirm that the details have been noted. "
        f"Here is a summary of the key points from your prompt:\n\n"
        f"{prompt}\n\n"
        f"Please let me know if you need any further assistance."
    )


def _openai_response(prompt: str, system_prompt: str, api_key: str | None, model: str | None = None) -> str:
    from openai import OpenAI
    config = get_config()
    target_model = model or config.llm_model or "gpt-3.5-turbo"
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=target_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def _anthropic_response(prompt: str, system_prompt: str, api_key: str | None, model: str | None = None) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise RuntimeError("pip install anthropic required for Anthropic support") from e
    
    config = get_config()
    target_model = model or config.llm_model or "claude-3-haiku-20240307"
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=target_model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.content[0].text


def _gemini_response(prompt: str, system_prompt: str, api_key: str | None, model: str | None = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is required for Gemini calls")
    
    config = get_config()
    target_model = model or config.llm_model or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={key}"
    
    payload = {
        "system_instruction": {"parts": {"text": system_prompt}},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3}
    }
    resp = httpx.post(url, json=payload, timeout=30.0)
    if resp.status_code != 200:
        print(f"❌ GEMINI API ERROR ({resp.status_code}):\n{resp.text}")
    
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError:
        return ""


def _ollama_response(prompt: str, system_prompt: str, api_base: str | None, model: str | None = None) -> str:
    config = get_config()
    base_url = api_base or os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    url = f"{base_url.rstrip('/')}/api/chat"
    
    target_model = model or config.llm_model or os.environ.get("OLLAMA_MODEL", "llama3")
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.3}
    }
    resp = httpx.post(url, json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def send_to_llm(prompt: str, strong_prompt: bool = False, api_key: str | None = None, model: str | None = None) -> str:
    """
    Send *prompt* to an LLM and return the response string.

    Uses config to determine the provider and strict mode settings.
    """
    config = get_config()

    if config.strict_mode or strong_prompt:
        system_prompt = get_preservation_prompt(strong=strong_prompt)
    else:
        system_prompt = "You are a helpful assistant."

    provider = config.llm_provider

    logger.info("Routing LLM call (provider=%s, model=%s, strict=%s)", provider, model or config.llm_model or "default", config.strict_mode)

    if provider == "openai":
        return _openai_response(prompt, system_prompt, api_key, model)
    elif provider == "anthropic":
        return _anthropic_response(prompt, system_prompt, api_key, model)
    elif provider == "gemini":
        return _gemini_response(prompt, system_prompt, api_key, model)
    elif provider == "ollama":
        return _ollama_response(prompt, system_prompt, config.llm_api_base, model)
    elif provider == "mock":
        return _mock_response(prompt)
    else:
        # Fallback to mock
        logger.warning(f"Unknown provider '{provider}', falling back to mock")
        return _mock_response(prompt)
