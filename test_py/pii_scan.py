"""Shared PII leak-detection scanner + the agreed sensitive-type registry.

This module is the single engine behind the PII no-leak tests (both the local
``tests/property/test_pii_no_leak_local.py`` suite and the deployed-stack e2e
``tests/e2e/test_scenarios_pii_security.py``). It provides:

  * ``SENSITIVE_TYPES`` — the SINGLE SOURCE OF TRUTH mapping each agreed
    sensitive data type -> its synthetic value. Add a field HERE (and to
    ``harness.SyntheticPII`` + the synthetic-document builder, which iterates
    this registry) and every positive/negative test and every surface scan
    covers it automatically.
  * ``mask`` — render a PII value so a failure diagnostic is TRACEABLE but never
    prints the raw value.
  * ``scan`` / ``assert_no_raw_pii`` — scan a serialized surface (logs, Step
    Functions events, LLM prompts, LLM responses, stored objects) for any raw
    sensitive value and, on a hit, raise a MASKED, traceable ``AssertionError``.
  * ``assert_pii_present`` — the positive-side assertion (a value MUST appear in
    the single authorized destination: the detokenized PEGA write output).

--- HOW TO ADD A SENSITIVE FIELD -------------------------------------------
  1. Add the value to ``tests/e2e/harness.py`` ``SyntheticPII`` (obviously
     synthetic; choose a value that cannot collide with non-PII document data).
  2. Add one line to ``SENSITIVE_TYPES`` below.
  3. The synthetic-document builders iterate ``SENSITIVE_TYPES``, so the doc
     picks it up automatically; the parametrized positive/negative tests and all
     surface scans then cover it with no further changes.
  If the new type's POSITIVE test then fails, the pipeline cannot yet
  detect/tokenize it — that failing test is the honest signal to extend the
  detectors (a separate, approved production change).
----------------------------------------------------------------------------

No real PII appears anywhere. Importing this module performs NO AWS calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from tests.synthetic_pii import SyntheticPII

# ---------------------------------------------------------------------------
# The agreed sensitive data types (extensible registry). type name -> synthetic
# value. This drives every scan and every parametrized positive/negative test.
# ---------------------------------------------------------------------------
SENSITIVE_TYPES: dict[str, str] = {
    "NAME": SyntheticPII.NAME,
    "ACCOUNT_ID": SyntheticPII.ACCOUNT_ID,
    "CARD_NUMBER": SyntheticPII.CARD_NUMBER,
    "SSN": SyntheticPII.SSN,
    "EMAIL": SyntheticPII.EMAIL,
    "PHONE": SyntheticPII.PHONE,
    "ADDRESS": SyntheticPII.ADDRESS,
    "DOB": SyntheticPII.DOB,
    "PASSPORT": SyntheticPII.PASSPORT,
    "DRIVERS_LICENSE": SyntheticPII.DRIVERS_LICENSE,
    "NATIONAL_ID": SyntheticPII.NATIONAL_ID,
}

# Types the current pipeline is NOT expected to detect/tokenize. The pipeline's
# tokenizer masks ANY value Comprehend flags: ``map_comprehend_type`` passes
# unrecognized Comprehend types (e.g. DATE_TIME for a DOB) THROUGH unchanged
# rather than dropping them, so they are tokenized + vaulted. ``_COMPREHEND_TYPE_MAP``
# only affects the vault *label* and the Comprehend stage's ``customer_info``
# field mapping — NOT whether a value is tokenized. So every agreed type is
# protected by the tokenizer as long as Comprehend flags it.
#
# CAVEAT (real-world recall, not a mechanism gap): whether real Comprehend
# reliably flags a BARE value (e.g. a lone "1970-01-01" DOB, or a country
# national-id format) as an entity depends on context/model recall. If the e2e
# suite shows a specific type surviving un-tokenized on the real stack, add it
# here to document that recall gap and drive a detector improvement.
KNOWN_DETECTION_GAPS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Hit:
    """A single detected raw-PII occurrence in a scanned surface."""

    type_name: str
    masked_value: str
    surface: str
    context: str  # snippet around the hit, with the raw value itself masked out


def mask(value: str) -> str:
    """Render ``value`` masked so a diagnostic is traceable but never raw.

    Reveals at most the first and last character (only when the value is long
    enough that this cannot re-identify a short secret); everything else becomes
    ``*``. Length is preserved so a reviewer can tell WHICH value matched without
    seeing it. Examples: ``"4111 1111 1111 1111"`` -> ``"4****************1"``;
    ``"000-00-0000"`` -> ``"0*********0"``; very short values are fully masked.
    """
    s = str(value)
    n = len(s)
    if n <= 4:
        return "*" * n
    return s[0] + ("*" * (n - 2)) + s[-1]


def _mask_in_context(haystack: str, value: str, radius: int = 24) -> str:
    """Return a short context snippet around the first hit, value masked out.

    The snippet lets a failure be TRACED to where in the surface the leak was,
    while the raw value is replaced by its masked form so the diagnostic itself
    never leaks PII.
    """
    idx = haystack.find(value)
    if idx == -1:
        return ""
    start = max(0, idx - radius)
    end = min(len(haystack), idx + len(value) + radius)
    snippet = haystack[start:end]
    # Redact the raw value from the snippet (there may be more than one).
    safe = snippet.replace(value, mask(value))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(haystack) else ""
    return f"{prefix}{safe}{suffix}"


def _to_text(haystack) -> str:
    """Coerce any surface (str / dict / list / object) to a scannable string.

    dicts/lists are JSON-serialized (default=str) so nested values are scanned
    too; anything else is ``str()``-ed.
    """
    if isinstance(haystack, str):
        return haystack
    if isinstance(haystack, (dict, list, tuple)):
        return json.dumps(haystack, default=str)
    return str(haystack)


def scan(haystack, values_by_type: dict[str, str] | None = None, *, surface: str = "") -> list[Hit]:
    """Scan ``haystack`` for any raw sensitive value; return the hits (masked).

    ``values_by_type`` defaults to the full ``SENSITIVE_TYPES`` registry.
    """
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


def assert_no_raw_pii(
    haystack,
    *,
    surface: str,
    values_by_type: dict[str, str] | None = None,
) -> None:
    """Fail (masked + traceable) if ANY raw sensitive value appears in ``haystack``.

    On a leak, raises ``AssertionError`` naming the SURFACE, the sensitive TYPE,
    the MASKED value, and a masked context snippet so the failure is diagnosable
    and traceable — WITHOUT the message ever containing the raw PII. Passes
    silently when the surface is clean.
    """
    hits = scan(haystack, values_by_type, surface=surface)
    if not hits:
        return
    lines = [f"Raw PII leaked into surface '{surface}': {len(hits)} hit(s)"]
    for h in hits:
        lines.append(
            f"  - type={h.type_name} masked_value={h.masked_value!r} "
            f"context={h.context!r}"
        )
    raise AssertionError("\n".join(lines))


def assert_pii_present(haystack, value: str, *, surface: str = "authorized_store") -> None:
    """Assert a value DOES appear (the single authorized destination — the write).

    Used on the detokenized PEGA write output to prove a handled value really was
    materialized at the one allowed exit point. The failure message masks the
    value so it never prints raw PII.
    """
    text = _to_text(haystack)
    assert value in text, (
        f"expected the (handled) value to be present in surface '{surface}' "
        f"(masked={mask(value)!r}) but it was absent"
    )
