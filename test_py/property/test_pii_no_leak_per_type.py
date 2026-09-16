"""Positive + negative PII no-leak coverage, PER agreed sensitive type.

Registry-driven: parametrized over ``tests.pii_scan.SENSITIVE_TYPES``, so adding
a type to the registry automatically adds its positive AND negative case here.

POSITIVE (per type): a document containing that type flows through the
AGENT-FREE span (tokenize -> preflight); the raw value must be vaulted (proof it
was HANDLED / tokenized) and must NOT appear in any non-authorized surface on
that span (logs / events / metadata / the tokens-only text). This uses
``_pii_harness.run_pipeline_pre_agent`` so it runs WITHOUT the agent — usable in
environments where the agent Lambda is not yet built. The full agent-span
no-leak assertion (adding the prompts/responses/write-sink surfaces) lives in
``test_pii_no_leak_local.py``.

NEGATIVE (per type): a deliberately-injected raw value of that type must be
CAUGHT by ``assert_no_raw_pii`` — proving the detector actually fires for every
type, with a masked, traceable diagnostic (and never printing the raw value).
These are PASSING tests that assert the leak IS detected; no real leak is left in
the pipeline. They exercise only the scanner, so they are fully agent-free.

The tokenizer masks ANY value Comprehend flags (unrecognized Comprehend types
like DATE_TIME for a DOB pass through ``map_comprehend_type`` unchanged and are
still tokenized/vaulted), so all 11 agreed types are protected by mechanism.
``KNOWN_DETECTION_GAPS`` is therefore empty; if the real-stack e2e run shows a
type surviving un-tokenized (a Comprehend recall gap on a bare value), add it to
that set and this suite will treat it accordingly.

No AWS. No real PII. NO AGENT DEPENDENCY.
"""
from __future__ import annotations

import json

import pytest

from tests.pii_scan import (
    KNOWN_DETECTION_GAPS,
    SENSITIVE_TYPES,
    assert_no_raw_pii,
    mask,
)
from tests.property._pii_harness import run_pipeline_pre_agent

# Protected types = every agreed type the pipeline is expected to tokenize today.
_PROTECTED_TYPES = [t for t in SENSITIVE_TYPES if t not in KNOWN_DETECTION_GAPS]


def _assert_clean_pre_agent_surfaces(caps, values_by_type):
    """Assert none of ``values_by_type`` appear in any non-authorized surface of
    the AGENT-FREE (tokenize -> preflight) span: events, logs, metadata, and the
    tokens-only text released downstream."""
    assert_no_raw_pii(caps["states"], surface="events", values_by_type=values_by_type)
    assert_no_raw_pii(caps["logs"], surface="logs", values_by_type=values_by_type)
    assert_no_raw_pii(caps["metadata"], surface="metadata", values_by_type=values_by_type)
    # The tokenized text handed downstream must carry tokens, not raw PII.
    assert_no_raw_pii(
        json.dumps(caps["tokenized_text"], default=str),
        surface="tokenized_text",
        values_by_type=values_by_type,
    )


# --- POSITIVE: each protected type is handled + absent from every surface ----


@pytest.mark.parametrize("type_name", _PROTECTED_TYPES)
def test_positive_type_is_tokenized_and_never_leaks(type_name):
    """A document containing ``type_name`` is tokenized/vaulted and leaks it
    nowhere on the agent-free (tokenize -> preflight) span."""
    value = SENSITIVE_TYPES[type_name]
    present = ["NAME"] if type_name == "NAME" else ["NAME", type_name]

    with run_pipeline_pre_agent(present) as caps:
        # (1) The raw value was HANDLED: it is in the vault (tokenized), proving
        #     the pipeline detected it rather than silently passing it through.
        assert value in caps["vault_values"], (
            f"{type_name} (masked={mask(value)!r}) was not tokenized/vaulted — "
            f"the pipeline does not protect this type"
        )
        # (2) The raw value leaks into NO non-authorized surface on this span.
        _assert_clean_pre_agent_surfaces(caps, {type_name: value})


# --- NEGATIVE: detection fires for each type (the safety net works) ----------


@pytest.mark.parametrize("type_name", list(SENSITIVE_TYPES))
def test_negative_injected_leak_is_detected(type_name):
    """A deliberately-planted raw value of ``type_name`` is CAUGHT (masked)."""
    value = SENSITIVE_TYPES[type_name]
    # Simulate a leak into a representative "events" surface (a stage return
    # dict) — the kind of thing that lands in Step Functions history.
    leaked_state = {"document_id": "doc-neg", "status": "succeeded", "leaked": value}
    with pytest.raises(AssertionError) as exc:
        assert_no_raw_pii(leaked_state, surface="events", values_by_type={type_name: value})
    msg = str(exc.value)
    # Traceable + masked: names the surface + type + masked value; NEVER raw.
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
