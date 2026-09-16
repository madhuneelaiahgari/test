"""Synthetic PII values used by the PII leak-detection tests.

Standalone (no AWS, no other imports) so the agent-free / AWS-free test core
(``pii_scan``, the local no-leak suites) does NOT depend on the e2e harness. The
e2e harness re-exports ``SyntheticPII`` from here for backward compatibility.

ALL VALUES ARE OBVIOUSLY SYNTHETIC. They contain NO real PII. Each is chosen so
it cannot collide with non-PII document text (e.g. the DOB differs from the
synthetic dispute/claim dates), so a leak of any value is unambiguous.
"""
from __future__ import annotations


class SyntheticPII:
    """Namespace of fixed, obviously-synthetic PII values for assertions."""

    NAME = "Jordan Sample"
    NAME_ALT = "Alex Placeholder"
    ACCOUNT_ID = "ACCT-0000-1111"
    ACCOUNT_ID_ALT = "ACCT-2222-3333"
    CARD_NUMBER = "4111 1111 1111 1111"  # canonical synthetic test PAN
    SSN = "000-00-0000"
    EMAIL = "jordan.sample@example.com"
    PHONE = "+1-555-0100"
    ADDRESS = "100 Test Street, Exampleville, EX 00000"
    # DOB is deliberately a value that cannot collide with the synthetic
    # dispute/claim date ("2024-01-03") used in documents, so a DOB leak is
    # unambiguous.
    DOB = "1970-01-01"
    PASSPORT = "X00000000"
    DRIVERS_LICENSE = "D0000-0000-0000"
    NATIONAL_ID = "NID-000-00-0000"  # non-SSN national-id form

    #: All synthetic PII literals, handy for "no raw PII leaked" scans.
    ALL = (
        NAME,
        NAME_ALT,
        ACCOUNT_ID,
        ACCOUNT_ID_ALT,
        CARD_NUMBER,
        SSN,
        EMAIL,
        PHONE,
        ADDRESS,
        DOB,
        PASSPORT,
        DRIVERS_LICENSE,
        NATIONAL_ID,
    )
