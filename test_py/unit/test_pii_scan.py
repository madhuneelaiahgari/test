"""Self-tests for the shared PII leak-detection scanner (tests/pii_scan.py).

These verify the DETECTOR itself before it is used to guard the pipeline:
  * ``mask`` never reveals more than the first/last char and never returns the
    raw value;
  * ``assert_no_raw_pii`` RAISES on a planted value, and its message is masked
    (contains the masked form, never the raw value);
  * it PASSES on token-only / clean text;
  * the registry holds every agreed sensitive type.
No AWS. No real PII.
"""
from __future__ import annotations

import pytest

from tests import pii_scan
from tests.pii_scan import SENSITIVE_TYPES, assert_no_raw_pii, mask, scan


# --- mask --------------------------------------------------------------------


def test_mask_hides_the_body_and_preserves_length():
    value = "4111 1111 1111 1111"
    masked = mask(value)
    assert masked != value
    assert len(masked) == len(value)
    # Only first + last char survive; everything else is masked.
    assert masked[0] == value[0]
    assert masked[-1] == value[-1]
    assert set(masked[1:-1]) == {"*"}


def test_mask_fully_masks_short_values():
    assert mask("abcd") == "****"
    assert mask("ab") == "**"
    assert mask("") == ""


def test_mask_never_returns_raw_value_for_every_registered_type():
    for value in SENSITIVE_TYPES.values():
        assert value not in mask(value) or len(value) <= 1


# --- registry ----------------------------------------------------------------


def test_registry_covers_all_agreed_types():
    expected = {
        "NAME", "ACCOUNT_ID", "CARD_NUMBER", "SSN", "EMAIL", "PHONE", "ADDRESS",
        "DOB", "PASSPORT", "DRIVERS_LICENSE", "NATIONAL_ID",
    }
    assert set(SENSITIVE_TYPES) == expected
    # Every registered value is a non-empty synthetic string.
    assert all(isinstance(v, str) and v for v in SENSITIVE_TYPES.values())


# --- assert_no_raw_pii: detection fires --------------------------------------


def test_assert_no_raw_pii_raises_on_planted_value():
    card = SENSITIVE_TYPES["CARD_NUMBER"]
    haystack = f"some log line containing {card} in the middle"
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(haystack, surface="logs")
    msg = str(exc.value)
    # Traceable: names the surface + the type.
    assert "logs" in msg
    assert "CARD_NUMBER" in msg
    # Masked: the raw value must NOT appear in the diagnostic.
    assert card not in msg
    # ... but the masked form does.
    assert mask(card) in msg


def test_assert_no_raw_pii_detects_in_nested_structures():
    # A dict/list surface is JSON-serialized and scanned recursively.
    name = SENSITIVE_TYPES["NAME"]
    surface = {"stage": "comprehend", "customer_info": {"name": name, "entities": [name]}}
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(surface, surface="events")
    assert name not in str(exc.value)
    assert "NAME" in str(exc.value)


def test_assert_no_raw_pii_reports_every_leaked_type():
    haystack = " ".join(SENSITIVE_TYPES.values())
    hits = scan(haystack)
    assert {h.type_name for h in hits} == set(SENSITIVE_TYPES)


# --- assert_no_raw_pii: clean surfaces pass ----------------------------------


def test_assert_no_raw_pii_passes_on_token_only_text():
    tokens = "TKN#NAME#0123456789ab TKN#ACCOUNT#abcdef012345 disputeType=fraud"
    # Should not raise.
    assert_no_raw_pii(tokens, surface="events")


def test_assert_no_raw_pii_passes_on_empty_and_scalar_surfaces():
    assert_no_raw_pii("", surface="logs")
    assert_no_raw_pii({"document_id": "abc", "status": "succeeded"}, surface="events")


# --- subset scanning (per-type) ----------------------------------------------


def test_scan_with_single_type_only_flags_that_type():
    haystack = SENSITIVE_TYPES["EMAIL"] + " and " + SENSITIVE_TYPES["PHONE"]
    hits = scan(haystack, {"EMAIL": SENSITIVE_TYPES["EMAIL"]})
    assert [h.type_name for h in hits] == ["EMAIL"]


def test_module_imports_without_aws():
    # Importing the scanner + harness performs no AWS calls (lazy clients).
    assert pii_scan.SENSITIVE_TYPES is SENSITIVE_TYPES
