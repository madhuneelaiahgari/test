"""Single-file PII no-leak test suite (drop-in for an auto-discovery CI).

WHAT THIS IS
------------
One self-contained pytest file that proves the IDP pipeline does not leak raw
PII. Copy this ONE file into your ``tests/`` folder; a pytest-based CI that
auto-discovers ``test_*.py`` runs it with no extra config, no subfolders, and no
sibling modules to import.

It bundles what were previously several modules:
  * the synthetic-PII registry (obviously fake values),
  * the masked leak-detection scanner (``mask`` / ``scan`` / ``assert_no_raw_pii``),
  * the in-memory AWS fakes + the AGENT-FREE pipeline harness
    (tokenize -> preflight), and
  * the test cases: scanner self-tests, per-type positive/negative, and the
    raw-stage / inline-envelope barrier tests.

WHY AGENT-FREE
--------------
Every test here runs WITHOUT the agent Lambda and WITHOUT AWS. It exercises the
single most important leak boundary — where raw PII is replaced by tokens and
vaulted (tokenize), then re-scanned by the gate (preflight) — plus the S3
hand-off barrier (``make_stage_output``). That covers the pipeline your customer
has deployed today (agent / pega_lookup are empty). The full agent-span test and
the deployed-stack e2e tests are intentionally NOT in this file (see "ADDING
MORE COVERAGE LATER" at the bottom).

WHAT THE CUSTOMER NEEDS
-----------------------
  * ``pytest`` (their CI already has it) + ``pip install ...`` nothing else here
    beyond the pipeline's own deps.
  * The pipeline source importable on ``sys.path`` — the same roots their other
    tests use to import ``handler_*`` / ``services`` (in our repo: ``src`` +
    ``src/python``). If ``pytest --collect-only`` errors on
    ``ModuleNotFoundError: handler_tokenizer`` etc., that path is not set.

NO REAL PII ANYWHERE. Importing this file performs NO AWS calls.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from unittest import mock

import pytest

# Pipeline code under test (real handlers + logic). These imports are the only
# coupling to the pipeline; they must resolve on the customer's test sys.path.
import services.common as common
import services.idp_logic as idp_logic
from handler_comprehend import main as comprehend
from handler_preflight import main as preflight
from handler_query_pega import main as query_pega
from handler_tokenizer import main as tokenizer


# ===========================================================================
# 1. Synthetic PII registry (SINGLE SOURCE OF TRUTH)
# ===========================================================================
# type name -> obviously-synthetic value. Add a field here and every scan +
# every parametrized positive/negative case below covers it automatically. Each
# value is chosen so it cannot collide with non-PII document text (e.g. the DOB
# differs from the synthetic dispute/claim dates), so a leak is unambiguous.

SENSITIVE_TYPES: dict[str, str] = {
    "NAME": "Jordan Sample",
    "ACCOUNT_ID": "ACCT-0000-1111",
    "CARD_NUMBER": "4111 1111 1111 1111",  # canonical synthetic test PAN
    "SSN": "000-00-0000",
    "EMAIL": "jordan.sample@example.com",
    "PHONE": "+1-555-0100",
    "ADDRESS": "100 Test Street, Exampleville, EX 00000",
    "DOB": "1970-01-01",
    "PASSPORT": "X00000000",
    "DRIVERS_LICENSE": "D0000-0000-0000",
    "NATIONAL_ID": "NID-000-00-0000",  # non-SSN national-id form
}

# Comprehend PII type per sensitive field, so the tokenizer's emitted token type
# matches. Unrecognized Comprehend types pass through map_comprehend_type
# unchanged and are still tokenized/vaulted, so this only affects the token
# LABEL, not whether a value is masked.
_COMPREHEND_TYPE = {
    "NAME": "NAME",
    "ACCOUNT_ID": "BANK_ACCOUNT_NUMBER",
    "CARD_NUMBER": "CREDIT_DEBIT_NUMBER",
    "SSN": "SSN",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "ADDRESS": "ADDRESS",
    "DOB": "DATE_TIME",
    "PASSPORT": "PASSPORT_NUMBER",
    "DRIVERS_LICENSE": "DRIVER_ID",
    "NATIONAL_ID": "SSN",
}

# Types the current pipeline is NOT expected to tokenize. The tokenizer masks
# ANY value Comprehend flags, so this is empty: a "gap" would be a Comprehend
# RECALL issue on a bare value (surfaced by the deployed-stack e2e run), not a
# mechanism gap. If the real stack shows a type surviving un-tokenized, add it
# here and its positive case is skipped accordingly.
KNOWN_DETECTION_GAPS: frozenset[str] = frozenset()


# ===========================================================================
# 2. Masked leak-detection scanner
# ===========================================================================


@dataclass(frozen=True)
class Hit:
    """A single detected raw-PII occurrence in a scanned surface."""

    type_name: str
    masked_value: str
    surface: str
    context: str  # snippet around the hit, with the raw value itself masked out


def mask(value: str) -> str:
    """Render ``value`` masked so a diagnostic is traceable but never raw.

    Reveals at most the first and last character (only when long enough that this
    cannot re-identify a short secret); everything else becomes ``*``. Length is
    preserved so a reviewer can tell WHICH value matched without seeing it.
    """
    s = str(value)
    n = len(s)
    if n <= 4:
        return "*" * n
    return s[0] + ("*" * (n - 2)) + s[-1]


def _mask_in_context(haystack: str, value: str, radius: int = 24) -> str:
    """Return a short context snippet around the first hit, value masked out."""
    idx = haystack.find(value)
    if idx == -1:
        return ""
    start = max(0, idx - radius)
    end = min(len(haystack), idx + len(value) + radius)
    snippet = haystack[start:end]
    safe = snippet.replace(value, mask(value))  # redact raw value from snippet
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(haystack) else ""
    return f"{prefix}{safe}{suffix}"


def _to_text(haystack) -> str:
    """Coerce any surface (str / dict / list / object) to a scannable string."""
    if isinstance(haystack, str):
        return haystack
    if isinstance(haystack, (dict, list, tuple)):
        return json.dumps(haystack, default=str)
    return str(haystack)


def scan(haystack, values_by_type: dict[str, str] | None = None, *, surface: str = "") -> list[Hit]:
    """Scan ``haystack`` for any raw sensitive value; return the hits (masked)."""
    values_by_type = values_by_type if values_by_type is not None else SENSITIVE_TYPES
    text = _to_text(haystack)
    hits: list[Hit] = []
    for type_name, value in values_by_type.items():
        if value and value in text:
            hits.append(
                Hit(
                    type_name=type_name,
                    masked_value=mask(value),
                    surface=surface,
                    context=_mask_in_context(text, value),
                )
            )
    return hits


def assert_no_raw_pii(haystack, *, surface: str, values_by_type: dict[str, str] | None = None) -> None:
    """Fail (masked + traceable) if ANY raw sensitive value appears in ``haystack``.

    On a leak, raises ``AssertionError`` naming the SURFACE, the sensitive TYPE,
    the MASKED value, and a masked context snippet — WITHOUT the message ever
    containing the raw PII. Passes silently when the surface is clean.
    """
    hits = scan(haystack, values_by_type, surface=surface)
    if not hits:
        return
    lines = [f"Raw PII leaked into surface '{surface}': {len(hits)} hit(s)"]
    for h in hits:
        lines.append(f"  - type={h.type_name} masked_value={h.masked_value!r} context={h.context!r}")
    raise AssertionError("\n".join(lines))


def assert_pii_present(haystack, value: str, *, surface: str = "authorized_store") -> None:
    """Assert a value DOES appear (the single authorized destination — the write)."""
    text = _to_text(haystack)
    assert value in text, (
        f"expected the (handled) value to be present in surface '{surface}' "
        f"(masked={mask(value)!r}) but it was absent"
    )


# ===========================================================================
# 3. In-memory AWS fakes + the AGENT-FREE pipeline harness
# ===========================================================================


class FakeVault:
    """In-memory PII_Vault: tokenizer writes, write stage reads via GSI query."""

    def __init__(self):
        self.items: list[dict] = []

    def put_item(self, Item):  # noqa: N803
        self.items.append(Item)
        return {}

    def query(self, **kwargs):
        doc = kwargs["ExpressionAttributeValues"][":doc"]
        return {"Items": [i for i in self.items if i.get("document_id") == doc]}


class FakeComprehend:
    """Flags exactly the given ``(value, comprehend_type)`` pairs by offset."""

    def __init__(self, pii_pairs):
        self.pii_pairs = pii_pairs

    def detect_pii_entities(self, Text, LanguageCode="en"):  # noqa: N803
        entities = []
        for value, ctype in self.pii_pairs:
            idx = Text.find(value)
            if idx == -1:
                continue
            entities.append(
                {"BeginOffset": idx, "EndOffset": idx + len(value), "Type": ctype, "Score": 0.99}
            )
        return {"Entities": entities}


class _MetadataCapture:
    """Capturing stand-in for ``record_metadata`` across stage modules."""

    def __init__(self):
        self.records: list[dict] = []

    def __call__(self, document_id, stage, status, *, reason=None,
                 missing_fields=None, start_ts=None, extra=None):
        self.records.append({
            "document_id": document_id, "stage": stage, "status": status,
            "reason": reason, "missing_fields": missing_fields,
            "start_ts": start_ts, "extra": extra,
        })


class _ListLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_metadata_and_logs():
    """Patch ``record_metadata`` on the tokenizer + preflight modules and capture
    root logs, so the harness can scan both surfaces for leaks."""
    capture = _MetadataCapture()
    log_handler = _ListLogHandler()
    root = logging.getLogger()
    prev_level = root.level
    with contextlib.ExitStack() as stack:
        for module in (tokenizer, preflight):
            stack.enter_context(mock.patch.object(module, "record_metadata", capture))
        root.addHandler(log_handler)
        root.setLevel(logging.DEBUG)
        try:
            yield capture, log_handler
        finally:
            root.removeHandler(log_handler)
            root.setLevel(prev_level)


@contextlib.contextmanager
def run_pipeline_pre_agent(present_types: list[str], *, document_id: str = "doc-preagent"):
    """Drive tokenize -> preflight ONLY (no agent). Agent-independent.

    ``present_types`` selects which SENSITIVE_TYPES appear in the document. Yields
    captured surfaces for scanning the tokens-only boundary:
      { states, logs, metadata, tokenized_text, vault_values, document_id }
    """
    lines = [SENSITIVE_TYPES[t] for t in present_types]
    structured_text = idp_logic.make_structured_text(document_id, printed=lines)
    pairs = [(SENSITIVE_TYPES[t], _COMPREHEND_TYPE[t]) for t in present_types]

    vault = FakeVault()
    states: list[dict] = []

    with _capture_metadata_and_logs() as (metadata, log_handler):
        # TOKENIZE — replaces raw PII with tokens; vaults token->PII mappings.
        tok = tokenizer.process(
            {"document_id": document_id, "structured_text": structured_text},
            comprehend_client=FakeComprehend(pairs),
            vault_table=vault,
        )
        states.append(tok)
        tokenized_text = tok.get("tokenized_text")

        # PREFLIGHT — inspects the tokenized text; empty comprehend => no
        # residual PII => gate passes and forwards the tokens-only text.
        pf = preflight.process(
            {"document_id": document_id, "tokenized_text": tokenized_text},
            comprehend_client=FakeComprehend([]),
            cloudwatch_client=None,
        )
        states.append(pf)

        yield {
            "document_id": document_id,
            "states": states,
            "logs": "\n".join(log_handler.messages),
            "metadata": metadata.records,
            "tokenized_text": tokenized_text,
            "vault_values": [i.get("pii_value") for i in vault.items],
        }


# ===========================================================================
# 4. TESTS — scanner self-tests (prove the detector before trusting it)
# ===========================================================================


def test_mask_hides_the_body_and_preserves_length():
    value = "4111 1111 1111 1111"
    masked = mask(value)
    assert masked != value
    assert len(masked) == len(value)
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


def test_registry_covers_all_agreed_types():
    expected = {
        "NAME", "ACCOUNT_ID", "CARD_NUMBER", "SSN", "EMAIL", "PHONE", "ADDRESS",
        "DOB", "PASSPORT", "DRIVERS_LICENSE", "NATIONAL_ID",
    }
    assert set(SENSITIVE_TYPES) == expected
    assert all(isinstance(v, str) and v for v in SENSITIVE_TYPES.values())


def test_assert_no_raw_pii_raises_on_planted_value():
    card = SENSITIVE_TYPES["CARD_NUMBER"]
    haystack = f"some log line containing {card} in the middle"
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(haystack, surface="logs")
    msg = str(exc.value)
    assert "logs" in msg
    assert "CARD_NUMBER" in msg
    assert card not in msg  # never the raw value
    assert mask(card) in msg  # but the masked form


def test_assert_no_raw_pii_detects_in_nested_structures():
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


def test_assert_no_raw_pii_passes_on_token_only_text():
    tokens = "TKN#NAME#0123456789ab TKN#ACCOUNT#abcdef012345 disputeType=fraud"
    assert_no_raw_pii(tokens, surface="events")  # must not raise


def test_assert_no_raw_pii_passes_on_empty_and_scalar_surfaces():
    assert_no_raw_pii("", surface="logs")
    assert_no_raw_pii({"document_id": "abc", "status": "succeeded"}, surface="events")


def test_scan_with_single_type_only_flags_that_type():
    haystack = SENSITIVE_TYPES["EMAIL"] + " and " + SENSITIVE_TYPES["PHONE"]
    hits = scan(haystack, {"EMAIL": SENSITIVE_TYPES["EMAIL"]})
    assert [h.type_name for h in hits] == ["EMAIL"]


# ===========================================================================
# 5. TESTS — per-type positive/negative (agent-free: tokenize -> preflight)
# ===========================================================================

_PROTECTED_TYPES = [t for t in SENSITIVE_TYPES if t not in KNOWN_DETECTION_GAPS]


def _assert_clean_pre_agent_surfaces(caps, values_by_type):
    """None of ``values_by_type`` appear in any non-authorized surface of the
    agent-free span: events, logs, metadata, and the tokens-only text."""
    assert_no_raw_pii(caps["states"], surface="events", values_by_type=values_by_type)
    assert_no_raw_pii(caps["logs"], surface="logs", values_by_type=values_by_type)
    assert_no_raw_pii(caps["metadata"], surface="metadata", values_by_type=values_by_type)
    assert_no_raw_pii(
        json.dumps(caps["tokenized_text"], default=str),
        surface="tokenized_text",
        values_by_type=values_by_type,
    )


@pytest.mark.parametrize("type_name", _PROTECTED_TYPES)
def test_positive_type_is_tokenized_and_never_leaks(type_name):
    """A document containing ``type_name`` is tokenized/vaulted and leaks it
    nowhere on the agent-free (tokenize -> preflight) span."""
    value = SENSITIVE_TYPES[type_name]
    present = ["NAME"] if type_name == "NAME" else ["NAME", type_name]

    with run_pipeline_pre_agent(present) as caps:
        # (1) The raw value was HANDLED: it is in the vault (tokenized).
        assert value in caps["vault_values"], (
            f"{type_name} (masked={mask(value)!r}) was not tokenized/vaulted — "
            f"the pipeline does not protect this type"
        )
        # (2) The raw value leaks into NO non-authorized surface on this span.
        _assert_clean_pre_agent_surfaces(caps, {type_name: value})


@pytest.mark.parametrize("type_name", list(SENSITIVE_TYPES))
def test_negative_injected_leak_is_detected(type_name):
    """A deliberately-planted raw value of ``type_name`` is CAUGHT (masked)."""
    value = SENSITIVE_TYPES[type_name]
    leaked_state = {"document_id": "doc-neg", "status": "succeeded", "leaked": value}
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(leaked_state, surface="events", values_by_type={type_name: value})
    msg = str(exc.value)
    assert "events" in msg
    assert type_name in msg
    assert value not in msg, "diagnostic must not contain the raw PII value"
    assert mask(value) in msg


@pytest.mark.parametrize("type_name", list(SENSITIVE_TYPES))
def test_negative_leak_in_prompt_is_detected(type_name):
    """A planted raw value in an LLM-prompt surface is CAUGHT (masked)."""
    value = SENSITIVE_TYPES[type_name]
    prompt = f"Classify this document. Customer detail: {value}. Return the tool."
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(prompt, surface="llm_prompt", values_by_type={type_name: value})
    assert value not in str(exc.value)
    assert "llm_prompt" in str(exc.value)


# ===========================================================================
# 6. TESTS — raw-PII stages + the inline-envelope barrier (make_stage_output)
# ===========================================================================
# The Comprehend stage produces customer_info with RAW PII (sliced from OCR
# text) and Query-PEGA carries it forward. That raw PII legitimately lives in
# the S3 STATE payload, but must NEVER reach the inline envelope that lands in
# Step Functions history. The barrier is common.make_stage_output, which keeps
# only SCALAR top-level fields inline and drops nested dicts/lists.

_ALL = dict(SENSITIVE_TYPES)


class _CapturingS3Service:
    """Stand-in for common._s3_service: records state + audit object writes."""

    def __init__(self):
        self.objects: dict[str, str] = {}

    def put_object(self, bucket, key, body, content_type=None):
        self.objects[key] = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else str(body)
        return {"ETag": "fake"}

    def get_object_bytes(self, bucket, key) -> bytes:
        return self.objects[key].encode("utf-8")

    def state_payload(self, stage: str) -> str:
        for key, body in self.objects.items():
            if key.startswith(f"state/{stage}/"):
                return body
        return ""


def test_make_stage_output_drops_nested_pii_keeps_scalars():
    """Nested PII-bearing values are dropped inline; only scalars survive."""
    ref = {"bucket": "state-bkt", "key": "state/comprehend/doc-1.json"}
    result = {
        "document_id": "doc-1",
        "comprehend_status": "succeeded",
        "has_customer_info": True,
        "customer_info": {
            "name": SENSITIVE_TYPES["NAME"],
            "account": SENSITIVE_TYPES["ACCOUNT_ID"],
            "entities": [{"type": "NAME", "text": SENSITIVE_TYPES["NAME"]}],
        },
        "structured_text": {"printed": [SENSITIVE_TYPES["CARD_NUMBER"]]},
    }
    envelope = common.make_stage_output(result, ref)

    assert envelope["state_ref"] == ref
    assert envelope["document_id"] == "doc-1"
    assert envelope["comprehend_status"] == "succeeded"
    assert envelope["has_customer_info"] is True
    assert "customer_info" not in envelope
    assert "structured_text" not in envelope
    assert_no_raw_pii(envelope, surface="events/inline_envelope", values_by_type=_ALL)


def test_make_stage_output_regression_scalar_pii_would_leak():
    """Guard: a raw PII value promoted to a top-level SCALAR WOULD ride inline —
    this asserts the scanner catches that regression."""
    ref = {"bucket": "b", "key": "state/comprehend/doc.json"}
    bad_result = {"document_id": "doc", "customerName": SENSITIVE_TYPES["NAME"]}
    envelope = common.make_stage_output(bad_result, ref)
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(envelope, surface="events/inline_envelope", values_by_type=_ALL)
    assert SENSITIVE_TYPES["NAME"] not in str(exc.value)
    assert mask(SENSITIVE_TYPES["NAME"]) in str(exc.value)


def test_make_stage_output_inline_passthrough_without_state_bucket():
    """Sanity: with NO state ref, make_stage_output returns the FULL result inline
    — documenting that the barrier depends on STATE_BUCKET being configured (the
    deployed stack sets it; the e2e suite verifies the real history is clean)."""
    result = {"document_id": "doc", "customer_info": {"name": SENSITIVE_TYPES["NAME"]}}
    envelope = common.make_stage_output(result, None)
    assert envelope is result


def _comprehend_service(fake_client):
    """Wrap a fake boto3-style comprehend client in a ComprehendService."""
    from services.comprehend_service import ComprehendService

    return ComprehendService(client=fake_client)


def _all_types_structured_text(document_id: str) -> dict:
    """OCR structured_text whose printed lines carry every synthetic value."""
    lines = [SENSITIVE_TYPES[t] for t in SENSITIVE_TYPES]
    return idp_logic.make_structured_text(document_id, printed=lines)


def test_comprehend_and_query_pega_inline_envelope_has_no_raw_pii(monkeypatch):
    """The Comprehend + Query-PEGA REAL handler() envelopes (what SFN history
    carries) contain no raw PII; the raw customer_info lands only in S3 state."""
    document_id = "doc-rawstage"
    fake_s3 = _CapturingS3Service()
    monkeypatch.setattr(common, "_s3_service", fake_s3)
    monkeypatch.setattr(common, "_STATE_BUCKET", "state-bkt")
    monkeypatch.setattr(common, "_AUDIT_BUCKET", "")  # skip audit writes here
    pairs = [(SENSITIVE_TYPES[t], _COMPREHEND_TYPE[t]) for t in SENSITIVE_TYPES]

    st = _all_types_structured_text(document_id)

    # --- Comprehend stage (real handler) ---
    comp_event = {"document_id": document_id, "structured_text": st}
    with mock.patch.object(
        comprehend, "ComprehendService", lambda client=None: _comprehend_service(FakeComprehend(pairs))
    ):
        comp_envelope = comprehend.handler(comp_event, None)

    assert_no_raw_pii(comp_envelope, surface="events/comprehend", values_by_type=_ALL)
    assert "state_ref" in comp_envelope
    comp_state = fake_s3.state_payload("comprehend")
    assert SENSITIVE_TYPES["NAME"] in comp_state, "customer_info should be in state payload"

    # --- Query-PEGA stage (real handler), fed the comprehend state payload ---
    qp_event = {"state_ref": comp_envelope["state_ref"]}
    qp_envelope = query_pega.handler(qp_event, None)

    assert_no_raw_pii(qp_envelope, surface="events/query_pega", values_by_type=_ALL)
    assert "state_ref" in qp_envelope
    qp_state = fake_s3.state_payload("query_pega")
    assert SENSITIVE_TYPES["NAME"] in qp_state


# ===========================================================================
# ADDING MORE COVERAGE LATER
# ===========================================================================
# This file is the AGENT-FREE tier (runs today with empty agent/pega_lookup
# Lambdas). Two further tiers exist in the full suite and can be added when the
# corresponding capability lands:
#
#   * FULL AGENT SPAN (local, no AWS): once the agent Lambda has real code, add a
#     run_pipeline that also drives agent -> case_router -> completeness ->
#     pega_write with a RecordingModel fake, and assert no raw PII in the LLM
#     prompt/response surfaces (the only surfaces the agent adds). The single
#     authorized exit — the detokenized PEGA write output — is the one place a
#     handled value may appear.
#
#   * DEPLOYED-STACK e2e (real AWS): upload an all-types synthetic doc, run the
#     real Step Functions execution, and scan the execution history + CloudWatch
#     logs + operational Metadata for raw PII (all 11 types), plus verify the
#     tokens-only vault entries and IAM explicit-deny on the PII vault. Gate the
#     whole module behind an opt-in env var (e.g. IDP_RUN_E2E=1) so an
#     auto-discovery CI SKIPS it without credentials rather than failing.
