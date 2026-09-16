"""No-leak coverage for the RAW-PII stages + the inline-envelope barrier.

The Comprehend stage produces ``customer_info`` containing RAW PII (name/account
sliced from the OCR text), and the Query-PEGA stage carries it forward. That raw
PII legitimately lives in the S3 STATE payload + the (encrypted) audit bucket —
but it must NEVER reach the inline envelope that lands in Step Functions
execution history. The single barrier enforcing that is
``common.make_stage_output``, which copies only SCALAR top-level fields inline
and drops nested dicts/lists (customer_info, structured_text, ...).

This module proves:
  1. ``make_stage_output`` drops nested PII-bearing values from the inline
     envelope (direct unit test + a regression guard).
  2. The REAL ``handler()`` path of the Comprehend and Query-PEGA stages emits a
     PII-free inline envelope, while the raw ``customer_info`` is written only to
     the S3 state payload.

No AWS. No real PII.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

import services.common as common
from handler_comprehend import main as comprehend
from handler_query_pega import main as query_pega

from tests.pii_scan import SENSITIVE_TYPES, assert_no_raw_pii, mask
# Import the fakes from the AGENT-FREE harness (not the agent-span test module),
# so this suite does not depend on the agent package.
from tests.property._pii_harness import FakeComprehend, _COMPREHEND_TYPE

# All raw synthetic values, keyed by type, for scanning.
_ALL = dict(SENSITIVE_TYPES)


# --- Fake state/audit S3 (captures every put_object the handler path makes) --


class _CapturingS3Service:
    """Stand-in for common._s3_service: records state + audit object writes."""

    def __init__(self):
        # key -> decoded body string
        self.objects: dict[str, str] = {}

    def put_object(self, bucket, key, body, content_type=None):  # noqa: N803-ish
        self.objects[key] = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else str(body)
        return {"ETag": "fake"}

    def get_object_bytes(self, bucket, key) -> bytes:
        """Read back a previously-written state object (SFN hand-off)."""
        return self.objects[key].encode("utf-8")

    def state_payload(self, stage: str) -> str:
        """Return the serialized state/{stage}/... object body (or '')."""
        for key, body in self.objects.items():
            if key.startswith(f"state/{stage}/"):
                return body
        return ""


# --- 1. make_stage_output: the inline-envelope barrier -----------------------


def test_make_stage_output_drops_nested_pii_keeps_scalars():
    """Nested PII-bearing values are dropped inline; only scalars survive."""
    ref = {"bucket": "state-bkt", "key": "state/comprehend/doc-1.json"}
    result = {
        "document_id": "doc-1",
        "comprehend_status": "succeeded",
        "has_customer_info": True,
        # RAW PII in a nested dict — must NOT appear in the inline envelope.
        "customer_info": {
            "name": SENSITIVE_TYPES["NAME"],
            "account": SENSITIVE_TYPES["ACCOUNT_ID"],
            "entities": [{"type": "NAME", "text": SENSITIVE_TYPES["NAME"]}],
        },
        "structured_text": {"printed": [SENSITIVE_TYPES["CARD_NUMBER"]]},
    }
    envelope = common.make_stage_output(result, ref)

    # Scalars + state_ref are kept inline.
    assert envelope["state_ref"] == ref
    assert envelope["document_id"] == "doc-1"
    assert envelope["comprehend_status"] == "succeeded"
    assert envelope["has_customer_info"] is True
    # Nested PII-bearing keys are dropped entirely.
    assert "customer_info" not in envelope
    assert "structured_text" not in envelope
    # And no raw PII survives anywhere in the inline envelope.
    assert_no_raw_pii(envelope, surface="events/inline_envelope", values_by_type=_ALL)


def test_make_stage_output_regression_scalar_pii_would_leak():
    """Guard: if a raw PII value were ever promoted to a top-level SCALAR, it
    WOULD ride inline — this asserts the scanner catches that regression."""
    ref = {"bucket": "b", "key": "state/comprehend/doc.json"}
    # Simulate a bad change that puts a raw value in a scalar field.
    bad_result = {"document_id": "doc", "customerName": SENSITIVE_TYPES["NAME"]}
    envelope = common.make_stage_output(bad_result, ref)
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(envelope, surface="events/inline_envelope", values_by_type=_ALL)
    assert SENSITIVE_TYPES["NAME"] not in str(exc.value)
    assert mask(SENSITIVE_TYPES["NAME"]) in str(exc.value)


def test_make_stage_output_inline_passthrough_without_state_bucket():
    """Sanity: with NO state ref (STATE_BUCKET unset) make_stage_output returns
    the FULL result inline — documenting that the barrier depends on the state
    bucket being configured (the deployed stack sets it)."""
    result = {"document_id": "doc", "customer_info": {"name": SENSITIVE_TYPES["NAME"]}}
    envelope = common.make_stage_output(result, None)
    # Full result inline -> PII WOULD be present. This is why the deployed stack
    # must set STATE_BUCKET (the e2e suite verifies the real history is clean).
    assert envelope is result


# --- 2. Real handler path: comprehend + query_pega inline envelope is clean --


def _all_types_structured_text(document_id: str) -> dict:
    """OCR structured_text whose printed lines carry every synthetic value."""
    import services.idp_logic as idp_logic

    lines = [SENSITIVE_TYPES[t] for t in SENSITIVE_TYPES]
    return idp_logic.make_structured_text(document_id, printed=lines)


def test_comprehend_and_query_pega_inline_envelope_has_no_raw_pii(monkeypatch):
    """The Comprehend + Query-PEGA REAL handler() envelopes (what SFN history
    carries) contain no raw PII; the raw customer_info lands only in S3 state."""
    document_id = "doc-rawstage"
    fake_s3 = _CapturingS3Service()
    # Make the handler path use the S3-handoff barrier with our capturing fake.
    monkeypatch.setattr(common, "_s3_service", fake_s3)
    monkeypatch.setattr(common, "_STATE_BUCKET", "state-bkt")
    monkeypatch.setattr(common, "_AUDIT_BUCKET", "")  # skip audit writes here
    # Comprehend flags every synthetic value so customer_info is fully populated.
    pairs = [(SENSITIVE_TYPES[t], _COMPREHEND_TYPE[t]) for t in SENSITIVE_TYPES]

    st = _all_types_structured_text(document_id)

    # --- Comprehend stage (real handler) ---
    comp_event = {"document_id": document_id, "structured_text": st}
    with mock.patch.object(
        comprehend, "ComprehendService", lambda client=None: _svc(FakeComprehend(pairs))
    ):
        comp_envelope = comprehend.handler(comp_event, None)

    # The inline envelope (SFN event) is PII-free ...
    assert_no_raw_pii(comp_envelope, surface="events/comprehend", values_by_type=_ALL)
    assert "state_ref" in comp_envelope
    # ... but the raw customer_info IS in the S3 state payload (authorized store).
    comp_state = fake_s3.state_payload("comprehend")
    assert SENSITIVE_TYPES["NAME"] in comp_state, "customer_info should be in state payload"

    # --- Query-PEGA stage (real handler), fed the comprehend state payload ---
    # Simulate the SFN hand-off: query_pega resolves the state_ref written above.
    qp_event = {"state_ref": comp_envelope["state_ref"]}
    qp_envelope = query_pega.handler(qp_event, None)

    assert_no_raw_pii(qp_envelope, surface="events/query_pega", values_by_type=_ALL)
    assert "state_ref" in qp_envelope
    # Query-PEGA carries customer_info forward -> present in ITS state payload too.
    qp_state = fake_s3.state_payload("query_pega")
    assert SENSITIVE_TYPES["NAME"] in qp_state


def _svc(fake_client):
    """Wrap a fake boto3-style comprehend client in a ComprehendService."""
    from services.comprehend_service import ComprehendService

    return ComprehendService(client=fake_client)
