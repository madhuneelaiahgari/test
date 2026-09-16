"""Shared harness for the local PII no-leak suites.

Holds the in-memory AWS fakes + metadata/log capture + the token helper used by
BOTH the agent-free tests and the full-agent-span test. Split out so the
AGENT-INDEPENDENT tests (per-type positive/negative, tokenize/preflight span)
can run WITHOUT importing the agent — important for environments where the agent
Lambda is not yet built/deployed (its handler code may be empty).

Two entry points:
  * ``run_pipeline_pre_agent`` (HERE) — drives tokenize -> preflight only. No
    agent import. Runs anywhere the tokenizer + preflight stage code exists.
  * ``run_pipeline`` (in ``test_pii_no_leak_local.py``) — the FULL span including
    the agent (tokenize -> preflight -> agent -> route -> completeness -> write).
    Reuses the fakes/capture from this module; only added when the agent exists.

No AWS. No real PII (all values are ``harness.SyntheticPII`` via SENSITIVE_TYPES).
"""
from __future__ import annotations

import contextlib
import logging
from unittest import mock

import services.idp_logic as idp_logic
from handler_preflight import main as preflight
from handler_tokenizer import main as tokenizer

from tests.pii_scan import SENSITIVE_TYPES

# Comprehend PII type per sensitive field, so the tokenizer's emitted token type
# matches. Types not natively self-mapping still tokenize (unrecognized
# Comprehend types pass through _COMPREHEND_TYPE_MAP unchanged), which is fine
# for a no-leak scan.
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


# --- In-memory AWS fakes -----------------------------------------------------


class FakeVault:
    """In-memory PII_Vault: tokenizer writes, write stage reads via GSI query."""

    def __init__(self):
        self.items: list[dict] = []

    def put_item(self, Item):  # noqa: N803
        self.items.append(Item)
        return {}

    def query(self, **kwargs):  # noqa: ANN003
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


class FakeS3:
    """Captures every put_object (the mock PEGA write sink + any state writes)."""

    def __init__(self):
        self.objects: dict[tuple[str, str], dict] = {}
        self.put_calls = 0

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.put_calls += 1
        self.objects[(Bucket, Key)] = {"Body": Body, "ContentType": ContentType}
        return {"ETag": "fake"}


class RecordingModel:
    """Fake agent ``invoke_model`` that RECORDS the prompt + response.

    Used only by the full-agent-span test. Kept here so both suites share it.
    ``invoke_model(prompt, tool) -> {classification, fields}``.
    """

    def __init__(self, classification: str, fields: dict):
        self._classification = classification
        self._fields = fields
        self.prompts: list[str] = []
        self.responses: list[dict] = []

    def __call__(self, prompt: str, tool: dict) -> dict:
        self.prompts.append(prompt)
        response = {"classification": self._classification, "fields": dict(self._fields)}
        self.responses.append(response)
        return response


def no_existing_case(_payload) -> list:
    """Agent PEGA lookup fake: never an existing case -> create route."""
    return []


class MetadataCapture:
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


class ListLogHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def capture_metadata_and_logs(extra_modules=()):
    """Patch ``record_metadata`` on the given handler modules + capture root logs.

    ``extra_modules`` lets the full-span test additionally patch the agent
    handler without this module importing the agent.
    """
    capture = MetadataCapture()
    log_handler = ListLogHandler()
    root = logging.getLogger()
    prev_level = root.level
    modules = (tokenizer, preflight) + tuple(extra_modules)
    with contextlib.ExitStack() as stack:
        for module in modules:
            stack.enter_context(mock.patch.object(module, "record_metadata", capture))
        root.addHandler(log_handler)
        root.setLevel(logging.DEBUG)
        try:
            yield capture, log_handler
        finally:
            root.removeHandler(log_handler)
            root.setLevel(prev_level)


def token_for(value: str, pii_type: str) -> str:
    """The canonical idp_logic token the tokenizer would emit for ``value``."""
    _, mapping = idp_logic.tokenize(value, [(value, pii_type)])
    (token,) = mapping.keys()
    return token


# --- Agent-FREE span: tokenize -> preflight ----------------------------------


@contextlib.contextmanager
def run_pipeline_pre_agent(present_types: list[str], *, document_id: str = "doc-preagent"):
    """Drive tokenize -> preflight ONLY (no agent). Agent-independent.

    ``present_types`` selects which SENSITIVE_TYPES appear in the document. Yields
    captured surfaces for scanning the tokens-only boundary:
      { states, logs, metadata, tokenized_text, vault_values, document_id }

    This covers the most important leak boundary — the point where raw PII is
    replaced with tokens and vaulted — for every sensitive type, WITHOUT needing
    the agent (or any downstream stage) to exist.
    """
    lines = [SENSITIVE_TYPES[t] for t in present_types]
    structured_text = idp_logic.make_structured_text(document_id, printed=lines)
    pairs = [(SENSITIVE_TYPES[t], _COMPREHEND_TYPE[t]) for t in present_types]

    vault = FakeVault()
    states: list[dict] = []

    with capture_metadata_and_logs() as (metadata, log_handler):
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
