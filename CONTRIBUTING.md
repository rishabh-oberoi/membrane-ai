# Contributing to Membrane

Thanks for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/rishabh-oberoi/Membrane.git
cd Membrane
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests must pass before submitting a PR.

## Project Structure

```
app/
├── config.py                     # Centralized configuration
├── pii/detector.py               # PII detection (Presidio + regex)
├── tokenizer/tokenizer.py        # Placeholder replacement
├── entity_tracker/tracker.py     # Context windows
├── entity_alignment/alignment.py # Identity restoration
├── llm/proxy.py                  # LLM integration
├── llm/integrity.py              # Placeholder validation + retry
├── rehydration/rehydrator.py     # Value restoration
├── audit.py                      # Audit logging
└── main.py                       # FastAPI application
```

## Guidelines

- **Keep it modular** — each module has a single responsibility
- **No LLM in alignment** — all alignment logic must be deterministic
- **Tests required** — every new feature needs test coverage
- **Run the full suite** before opening a PR

## Adding a New Entity Type

1. Add detection patterns in `app/pii/detector.py`
2. Add alias list in `app/entity_alignment/alignment.py`
3. Add paraphrase patterns in `app/rehydration/rehydrator.py`
4. Add tests in `tests/test_pipeline.py`

## Code Style

- Python 3.11+
- Type hints on all public functions
- Docstrings on modules and public functions
- `from __future__ import annotations` in every module
