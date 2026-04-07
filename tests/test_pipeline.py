"""
Tests — Trust Layer for LLMs (v3)
==================================
Comprehensive suite including Entity Alignment Layer tests.
"""

from __future__ import annotations

import pytest

from membrane.config import get_config, reset_config
from membrane.pii.detector import detect_pii, PIIEntity, _regex_detect
from membrane.tokenizer.tokenizer import tokenize
from membrane.entity_tracker.tracker import EntityTracker, _extract_context
from membrane.entity_alignment.alignment import align_entities, AlignmentResult
from membrane.rehydration.rehydrator import rehydrate
from membrane.llm.proxy import send_to_llm
from membrane.llm.integrity import check_placeholders, send_to_llm_with_retry, LLMResult, IntegrityResult
from membrane.audit import log_request, get_logs, clear_logs


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

class TestConfig:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_defaults(self):
        c = get_config()
        assert c.enable_presidio is True
        assert c.max_retries == 1

    def test_frozen(self):
        with pytest.raises(AttributeError):
            get_config().strict_mode = False  # type: ignore


# ══════════════════════════════════════════════════════════════
# Entity Tracker
# ══════════════════════════════════════════════════════════════

class TestEntityTracker:
    def test_context_extraction(self):
        before, after = _extract_context("Hello world John Doe is here", 12, 20, 3)
        assert len(before) > 0 or len(after) > 0

    def test_track_and_retrieve(self):
        t = EntityTracker()
        t.track("PERSON_1", "John", "PERSON", 0.85, 0, 4, "John went home")
        assert t.get("PERSON_1") is not None
        assert len(t.context_words("PERSON_1")) > 0

    def test_to_dict(self):
        t = EntityTracker()
        t.track("PERSON_1", "Alice", "PERSON", 0.9, 0, 5, "Alice went home")
        d = t.to_dict()
        assert d["PERSON_1"]["value"] == "Alice"


# ══════════════════════════════════════════════════════════════
# PII Detection
# ══════════════════════════════════════════════════════════════

class TestPIIDetection:
    def test_detect_email(self):
        entities = detect_pii("Contact info@example.com please.")
        assert any(e.entity_type == "EMAIL" for e in entities)

    def test_detect_name(self):
        entities = detect_pii("John Doe arrived today.")
        assert any(e.entity_type == "PERSON" for e in entities)

    def test_sorted_by_position(self):
        entities = detect_pii("Alice at alice@test.com called 555-111-2222.")
        assert [e.start for e in entities] == sorted(e.start for e in entities)


class TestRegexFallback:
    def test_email(self):
        assert any(e.entity_type == "EMAIL" for e in _regex_detect("john@example.com"))

    def test_phone(self):
        assert any(e.entity_type == "PHONE" for e in _regex_detect("Call (800) 555-0199"))

    def test_name(self):
        assert any(e.entity_type == "PERSON" for e in _regex_detect("Contact John Doe"))

    def test_ssn(self):
        assert any(e.entity_type == "SSN" for e in _regex_detect("SSN is 123-45-6789"))

    def test_credit_card(self):
        assert any(e.entity_type == "CREDIT_CARD" for e in _regex_detect("Card 4111-1111-1111-1111"))

    def test_dob(self):
        assert any(e.entity_type == "DATE_OF_BIRTH" for e in _regex_detect("DOB: 03/15/1990"))

    def test_dob_written(self):
        assert any(e.entity_type == "DATE_OF_BIRTH" for e in _regex_detect("Born on January 5, 1990"))

    def test_url(self):
        assert any(e.entity_type == "URL" for e in _regex_detect("Visit https://example.com/page"))


# ══════════════════════════════════════════════════════════════
# Tokenization
# ══════════════════════════════════════════════════════════════

class TestTokenization:
    def test_basic(self):
        r = tokenize("Hello John", [PIIEntity("John", "PERSON", 6, 10, 0.85)])
        assert "<PERSON_1>" in r.anonymized_text
        assert r.mapping["PERSON_1"]["value"] == "John"

    def test_tracker_populated(self):
        r = tokenize("Hello John", [PIIEntity("John", "PERSON", 6, 10, 0.85)])
        assert len(r.tracker) == 1

    def test_context_in_mapping(self):
        """Mapping should include context words from the tracker."""
        r = tokenize("Hello John Doe today", [PIIEntity("John Doe", "PERSON", 6, 14, 0.85)])
        assert "context" in r.mapping["PERSON_1"]
        assert isinstance(r.mapping["PERSON_1"]["context"], list)

    def test_dedup(self):
        r = tokenize("Alice met Alice", [
            PIIEntity("Alice", "PERSON", 0, 5, 0.85),
            PIIEntity("Alice", "PERSON", 10, 15, 0.85),
        ])
        assert r.anonymized_text.count("<PERSON_1>") == 2

    def test_ssn_tokenization(self):
        r = tokenize("SSN: 123-45-6789", [PIIEntity("123-45-6789", "SSN", 5, 16, 0.9)])
        assert "<SSN_1>" in r.anonymized_text
        assert r.mapping["SSN_1"]["value"] == "123-45-6789"

    def test_credit_card_tokenization(self):
        r = tokenize("Card: 4111-1111-1111-1111", [PIIEntity("4111-1111-1111-1111", "CREDIT_CARD", 6, 25, 0.85)])
        assert "<CREDIT_CARD_1>" in r.anonymized_text

    def test_url_tokenization(self):
        r = tokenize("Visit https://example.com", [PIIEntity("https://example.com", "URL", 6, 25, 0.9)])
        assert "<URL_1>" in r.anonymized_text


# ══════════════════════════════════════════════════════════════
# Entity Alignment ← NEW (Key Tests)
# ══════════════════════════════════════════════════════════════

class TestEntityAlignment:
    """Tests for the Entity Alignment Layer."""

    def test_patient_to_person(self):
        """'the patient' → John Doe"""
        mapping = {"PERSON_1": {"value": "John Doe", "type": "PERSON", "context": ["emailed", "clinic"]}}
        result = align_entities("The patient contacted the clinic", mapping)
        assert result.aligned is True
        assert "John Doe" in result.text
        assert "patient" not in result.text.lower()
        assert result.confidence >= 0.7

    def test_customer_to_person(self):
        """'the customer' → John Doe"""
        mapping = {"PERSON_1": {"value": "John Doe", "type": "PERSON", "context": ["called", "invoice"]}}
        result = align_entities("The customer called about the invoice", mapping)
        assert result.aligned is True
        assert "John Doe" in result.text

    def test_email_alias(self):
        """'the email address' → actual email"""
        mapping = {"EMAIL_1": {"value": "john@example.com", "type": "EMAIL", "context": ["send", "report"]}}
        result = align_entities("Send the report to the email address", mapping)
        assert result.aligned is True
        assert "john@example.com" in result.text

    def test_no_replacement_when_placeholder_present(self):
        """If placeholder is already there, skip alignment."""
        mapping = {"PERSON_1": {"value": "John", "type": "PERSON", "context": []}}
        result = align_entities("Hello <PERSON_1>, how are you?", mapping)
        assert result.aligned is False
        assert result.confidence == 1.0

    def test_no_replacement_when_ambiguous(self):
        """Multiple entities of same type → no replacement (ambiguous)."""
        mapping = {
            "PERSON_1": {"value": "Alice", "type": "PERSON", "context": []},
            "PERSON_2": {"value": "Bob", "type": "PERSON", "context": []},
        }
        # Both placeholders missing, but two PERSON entities → ambiguous
        result = align_entities("The patient visited the clinic", mapping)
        assert result.aligned is False

    def test_no_alias_found(self):
        """No alias words in LLM output → no replacement."""
        mapping = {"PERSON_1": {"value": "John", "type": "PERSON", "context": []}}
        result = align_entities("The weather is sunny today", mapping)
        assert result.aligned is False

    def test_empty_mapping(self):
        result = align_entities("Hello world", {})
        assert result.aligned is False
        assert result.confidence == 1.0

    def test_replacement_metadata(self):
        """Check that replacement details are recorded."""
        mapping = {"PERSON_1": {"value": "Jane", "type": "PERSON", "context": ["called"]}}
        result = align_entities("The client called yesterday", mapping)
        assert len(result.replacements) == 1
        assert result.replacements[0].alias == "client"
        assert result.replacements[0].original_value == "Jane"
        assert result.replacements[0].placeholder_key == "PERSON_1"

    def test_context_boosts_confidence(self):
        """Context overlap should increase confidence."""
        mapping_with_ctx = {
            "PERSON_1": {"value": "John", "type": "PERSON",
                         "context": ["emailed", "clinic", "appointment"]},
        }
        mapping_no_ctx = {
            "PERSON_1": {"value": "John", "type": "PERSON", "context": []},
        }
        text = "The patient emailed the clinic about the appointment"
        r1 = align_entities(text, mapping_with_ctx)
        r2 = align_entities(text, mapping_no_ctx)
        assert r1.confidence >= r2.confidence

    # ── NEW: Extended entity type alignment tests ──

    def test_ssn_alias(self):
        """'the social security number' → actual SSN"""
        mapping = {"SSN_1": {"value": "123-45-6789", "type": "SSN", "context": ["verify"]}}
        result = align_entities("Please verify the social security number", mapping)
        assert result.aligned is True
        assert "123-45-6789" in result.text

    def test_credit_card_alias(self):
        """'the credit card' → actual card number"""
        mapping = {"CREDIT_CARD_1": {"value": "4111-1111-1111-1111", "type": "CREDIT_CARD", "context": ["charge"]}}
        result = align_entities("Charge the credit card for the purchase", mapping)
        assert result.aligned is True
        assert "4111-1111-1111-1111" in result.text

    def test_location_alias(self):
        """'the address' → actual location"""
        mapping = {"LOCATION_1": {"value": "123 Main St, Springfield", "type": "LOCATION", "context": ["ship"]}}
        result = align_entities("Ship the package to the address", mapping)
        assert result.aligned is True
        assert "123 Main St, Springfield" in result.text

    def test_dob_alias(self):
        """'the date of birth' → actual DOB"""
        mapping = {"DATE_OF_BIRTH_1": {"value": "03/15/1990", "type": "DATE_OF_BIRTH", "context": ["patient"]}}
        result = align_entities("The patient's date of birth was confirmed", mapping)
        assert result.aligned is True
        assert "03/15/1990" in result.text

    def test_url_alias(self):
        """'the website' → actual URL"""
        mapping = {"URL_1": {"value": "https://example.com", "type": "URL", "context": ["visit"]}}
        result = align_entities("Please visit the website for details", mapping)
        assert result.aligned is True
        assert "https://example.com" in result.text


# ══════════════════════════════════════════════════════════════
# Integrity
# ══════════════════════════════════════════════════════════════

class TestIntegrity:
    def test_all_present(self):
        r = check_placeholders("Hi <PERSON_1>", {"PERSON_1": {"value": "J", "type": "PERSON"}})
        assert r.score == 1.0

    def test_missing(self):
        r = check_placeholders("Hi there", {"PERSON_1": {"value": "J", "type": "PERSON"}})
        assert r.score == 0.0

    def test_retry_mock(self):
        result = send_to_llm_with_retry("<PERSON_1> is here",
                                         {"PERSON_1": {"value": "John", "type": "PERSON"}})
        assert result.status == "ok"

    def test_structured_error(self):
        r = LLMResult(
            response="no placeholders",
            integrity=IntegrityResult(total=2, preserved=0, missing=["PERSON_1", "EMAIL_1"], score=0.0),
            retry_count=1, status="failed",
            error={"code": "PLACEHOLDER_LOSS", "missing_tokens": ["<PERSON_1>"]},
        )
        assert r.error["code"] == "PLACEHOLDER_LOSS"


# ══════════════════════════════════════════════════════════════
# Rehydration
# ══════════════════════════════════════════════════════════════

class TestRehydration:
    def test_exact(self):
        r, a = rehydrate("Hello <PERSON_1>", {"PERSON_1": {"value": "John", "type": "PERSON"}})
        assert r == "Hello John"

    def test_paraphrase(self):
        r, _ = rehydrate("Notify the patient.", {"PERSON_1": {"value": "Alice", "type": "PERSON"}})
        assert "Alice" in r

    def test_no_replace_ambiguous(self):
        r, _ = rehydrate("Notify the person.", {
            "PERSON_1": {"value": "A", "type": "PERSON"},
            "PERSON_2": {"value": "B", "type": "PERSON"},
        })
        assert "the person" in r

    def test_backward_compat_no_tracker(self):
        r, _ = rehydrate("Send to <EMAIL_1>", {"EMAIL_1": {"value": "x@y.com", "type": "EMAIL"}})
        assert "x@y.com" in r

    # ── NEW: Extended entity type rehydration tests ──

    def test_ssn_exact(self):
        r, _ = rehydrate("SSN is <SSN_1>", {"SSN_1": {"value": "123-45-6789", "type": "SSN"}})
        assert "123-45-6789" in r

    def test_ssn_paraphrase(self):
        r, _ = rehydrate("Verify the SSN.", {"SSN_1": {"value": "123-45-6789", "type": "SSN"}})
        assert "123-45-6789" in r

    def test_credit_card_exact(self):
        r, _ = rehydrate("Card: <CREDIT_CARD_1>", {"CREDIT_CARD_1": {"value": "4111-1111-1111-1111", "type": "CREDIT_CARD"}})
        assert "4111-1111-1111-1111" in r

    def test_url_exact(self):
        r, _ = rehydrate("Visit <URL_1>", {"URL_1": {"value": "https://example.com", "type": "URL"}})
        assert "https://example.com" in r


# ══════════════════════════════════════════════════════════════
# Audit Logging
# ══════════════════════════════════════════════════════════════

class TestAuditLogging:
    def setup_method(self):
        clear_logs()

    def test_log_and_retrieve(self):
        log_request("a", "b", "c", "d", {}, 1.0, 0, "ok")
        assert len(get_logs()) == 1

    def test_clear(self):
        log_request("x", "x", "x", "x", {}, 1.0, 0, "ok")
        assert clear_logs() == 1


# ══════════════════════════════════════════════════════════════
# Full Pipeline (with alignment)
# ══════════════════════════════════════════════════════════════

class TestPipelineRoundTrip:
    def test_standard_round_trip(self):
        """Standard pipeline: detect → tokenize → LLM → align → rehydrate."""
        original = "Contact John Doe at john@example.com."
        entities = detect_pii(original)
        result = tokenize(original, entities)

        assert "john@example.com" not in result.anonymized_text

        llm_result = send_to_llm_with_retry(result.anonymized_text, result.mapping)
        alignment = align_entities(llm_result.response, result.mapping)
        final, _ = rehydrate(alignment.text, result.mapping, tracker=result.tracker)
        assert "john@example.com" in final

    def test_alignment_round_trip(self):
        """Pipeline with simulated paraphrase: alignment should restore identity."""
        original = "John Doe emailed the clinic."
        entities = detect_pii(original)
        result = tokenize(original, entities)

        # Simulate LLM paraphrasing away the placeholder
        paraphrased = "The patient contacted the clinic for an appointment."

        alignment = align_entities(paraphrased, result.mapping)
        final, _ = rehydrate(alignment.text, result.mapping, tracker=result.tracker)
        assert "John Doe" in final

    def test_extended_entity_round_trip(self):
        """Pipeline with SSN and credit card — detect, tokenize, rehydrate."""
        # Test SSN detection and round-trip
        ssn_text = "The SSN is 123-45-6789 for the patient."
        ssn_entities = _regex_detect(ssn_text)
        has_ssn = any(e.entity_type == "SSN" for e in ssn_entities)
        assert has_ssn, f"Expected SSN, got: {[(e.entity_type, e.value) for e in ssn_entities]}"
        ssn_result = tokenize(ssn_text, ssn_entities)
        assert "123-45-6789" not in ssn_result.anonymized_text
        final_ssn, _ = rehydrate(ssn_result.anonymized_text, ssn_result.mapping, tracker=ssn_result.tracker)
        assert "123-45-6789" in final_ssn

        # Test credit card detection and round-trip
        cc_text = "Card number 4111-1111-1111-1111 on file."
        cc_entities = _regex_detect(cc_text)
        has_cc = any(e.entity_type == "CREDIT_CARD" for e in cc_entities)
        assert has_cc, f"Expected CREDIT_CARD, got: {[(e.entity_type, e.value) for e in cc_entities]}"
        cc_result = tokenize(cc_text, cc_entities)
        assert "4111-1111-1111-1111" not in cc_result.anonymized_text
        final_cc, _ = rehydrate(cc_result.anonymized_text, cc_result.mapping, tracker=cc_result.tracker)
