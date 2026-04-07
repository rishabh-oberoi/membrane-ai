#!/usr/bin/env python3
"""
Membrane AI — Enterprise Trust Layer CLI (v3)
=============================================
A professional demonstration of reversible PII anonymization and 
entity alignment for secure LLM workflows.

Usage:
    python examples/cli.py "John Doe emailed john@example.com"
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

# Ensure the parent directory is in the path so we can import 'membrane'
# even when running from within the /examples folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from membrane.pii.detector import detect_pii
from membrane.tokenizer.tokenizer import tokenize
from membrane.llm.integrity import send_to_llm_with_retry
from membrane.entity_alignment.alignment import align_entities
from membrane.rehydration.rehydrator import rehydrate

def _sep(title: str) -> str:
    return f"\n{'─' * 60}\n  {title}\n{'─' * 60}"

def run_pipeline(prompt: str, mock_llm_response: str | None = None) -> None:
    """Execute the full pipeline with latency tracking."""
    
    start_time = time.perf_counter()
    print(_sep("1 │ ORIGINAL PROMPT"))
    print(prompt)

    # 1. Detect PII
    entities = detect_pii(prompt)
    print(_sep("2 │ DETECTED PII ENTITIES"))
    if entities:
        for e in entities:
            print(f"  • [{e.entity_type}] \"{e.value}\" "
                  f"(chars {e.start}–{e.end}, confidence={e.confidence})")
    else:
        print("  (none detected)")

    # 2. Tokenize
    result = tokenize(prompt, entities)
    print(_sep("3 │ ANONYMIZED PROMPT"))
    print(result.anonymized_text)

    print(_sep("4 │ PII MAPPING (Secure Local Store)"))
    print(json.dumps(result.mapping, indent=2))

    # 3. LLM call
    if mock_llm_response is not None:
        llm_response_text = mock_llm_response
        print(_sep("5 │ SIMULATED LLM RESPONSE"))
        print(llm_response_text)
    else:
        llm_result = send_to_llm_with_retry(result.anonymized_text, result.mapping)
        llm_response_text = llm_result.response
        print(_sep("5 │ RAW LLM RESPONSE"))
        print(llm_response_text)
        print(_sep("6 │ INTEGRITY METRICS"))
        print(f"  Placeholders total:     {llm_result.integrity.total}")
        print(f"  Placeholders preserved: {llm_result.integrity.preserved}")
        print(f"  Integrity score:        {llm_result.integrity.score}")
        print(f"  Status:                 {llm_result.status}")
        # Hint at Enterprise features
        # print("  Enterprise Audit Log:  [Available in Membrane Enterprise]")

    # 4. Entity Alignment
    alignment = align_entities(llm_response_text, result.mapping)
    print(_sep("7 │ ENTITY ALIGNMENT (Semantic Check)"))
    if alignment.aligned:
        print(f"  ✅ Aligned: {len(alignment.replacements)} replacement(s)")
        for r in alignment.replacements:
            print(f"    • \"{r.alias}\" → \"{r.original_value}\"")
        print(f"\n  Aligned text:\n  {alignment.text}")
    else:
        print("  (no alignment needed — placeholders intact)")

    # 5. Rehydration
    final, actions = rehydrate(
        alignment.text, result.mapping, tracker=result.tracker,
    )
    
    end_time = time.perf_counter()
    latency = end_time - start_time

    print(_sep("8 │ FINAL REHYDRATED RESPONSE"))
    print(final)

    if actions:
        print(_sep("9 │ REHYDRATION ACTIONS"))
        for target, method in actions.items():
            print(f"  • {target} → {method}")

    print(f"\n{'═' * 60}")
    print(f"  ✅ Pipeline complete in {latency:.4f}s")
    print(f"{'═' * 60}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Membrane AI — Enterprise Trust Layer CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt text to process")
    args = parser.parse_args()

    if args.prompt:
        run_pipeline(args.prompt)
    else:
        print("Membrane AI CLI — Professional Preview\n")
        print("Enter a prompt containing PII (e.g., names or emails) to see the trust layer in action.")
        try:
            while True:
                text = input("\n>>> ").strip()
                if text:
                    run_pipeline(text)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. For production support, visit arogyam-health.com/membrane")
            sys.exit(0)

if __name__ == "__main__":
    main()