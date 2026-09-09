 have all the current content. Here are the copy-ready blocks for each file, showing exactly what to add/change. Everything below matches your workspace's line-only state.

1. 
textract_service.py
Replace the whole file (it's small, and the module docstring + methods all changed):

python

"""Amazon Textract wrapper — ASYNCHRONOUS TEXT DETECTION for the OCR stage.

The OCR stage uses Textract's ASYNC TEXT-DETECTION API
(``StartDocumentTextDetection`` / ``GetDocumentTextDetection``), so multi-page
PDFs/TIFFs are supported. Text detection returns LINE + WORD blocks ONLY (no
KEY_VALUE_SET/CELL) — OCR is used purely for text EXTRACTION and the LINE view is
the single source of truth for every downstream stage (Comprehend, Tokenizer,
Agent). The analysis API (``StartDocumentAnalysis`` + FORMS/TABLES) was
intentionally dropped: its structural views are unreliable on scanned/handwritten
documents and diverge from the LINE view.

The async job is orchestrated by Step Functions: the OCR (start) stage kicks off
the job and returns its ``JobId``; a Wait + collect loop polls until it is done.
"""
from __future__ import annotations

import os

import boto3

# Textract async ``GetDocumentTextDetection`` JobStatus values (Req 2.x).
JOB_IN_PROGRESS = "IN_PROGRESS"
JOB_SUCCEEDED = "SUCCEEDED"
JOB_FAILED = "FAILED"
JOB_PARTIAL_SUCCESS = "PARTIAL_SUCCESS"

class TextractService:
    """Thin, consistent wrapper around the Textract client."""

    def __init__(self, client=None, region_name=None):
        self._client = client
        # Explicit region (defaults to the Lambda runtime's AWS_REGION). Textract
        # is a regional service; pinning the region makes the regional endpoint
        # (incl. an interface VPC endpoint via Private DNS) unambiguous.
        self._region = region_name or os.environ.get("AWS_REGION")

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client("textract", region_name=self._region)
        return self._client

    # --- Async (multi-page) TEXT DETECTION -----------------------------------
    #
    # Text-only OCR: ``StartDocumentTextDetection`` / ``GetDocumentTextDetection``
    # return LINE + WORD blocks ONLY (no KEY_VALUE_SET/CELL). This is the correct
    # API for "OCR just for extraction" — it has no ``FeatureTypes`` parameter and
    # cannot produce forms/tables, so the LINE view is the single source of truth
    # downstream.

    def start_document_text_detection(self, bucket: str, key: str) -> str:
        """``StartDocumentTextDetection`` on an S3 object; returns the ``JobId``.

        Kicks off an asynchronous, multi-page TEXT-ONLY OCR job (LINE/WORD only;
        no forms/tables). Returns immediately with the job id — results are
        fetched later via :meth:`get_document_text_detection`.
        """
        response = self.client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        )
        return response["JobId"]

    def get_document_text_detection(self, job_id: str) -> dict:
        """Fetch a text-detection job's status and (when done) ALL of its Blocks.

        Returns ``{"status", "blocks", "status_message"}``. ``IN_PROGRESS`` /
        ``FAILED`` return empty ``blocks``; ``SUCCEEDED`` / ``PARTIAL_SUCCESS``
        page through every ``NextToken`` so the caller gets the complete LINE/WORD
        block list in one shape. On ``FAILED`` the ``status_message`` carries
        Textract's reason.
        """
        first = self.client.get_document_text_detection(JobId=job_id)
        status = first.get("JobStatus", "")
        result = {
            "status": status,
            "blocks": [],
            "status_message": first.get("StatusMessage", ""),
        }
        if status == JOB_IN_PROGRESS or status == JOB_FAILED:
            return result

        # SUCCEEDED / PARTIAL_SUCCESS: page through every NextToken.
        blocks = list(first.get("Blocks", []) or [])
        next_token = first.get("NextToken")
        while next_token:
            page = self.client.get_document_text_detection(
                JobId=job_id, NextToken=next_token
            )
            blocks.extend(page.get("Blocks", []) or [])
            next_token = page.get("NextToken")
        result["blocks"] = blocks
        return result
Removed vs their old version: DEFAULT_FEATURE_TYPES, start_document_analysis, get_document_analysis, analyze_document.

2. 
main.py
Import line — change:

python

# OLD
from services.textract_service import DEFAULT_FEATURE_TYPES, TextractService
# NEW
from services.textract_service import TextractService
In process(...) — change the start call:

python

# OLD
    try:
        job_id = textract.start_document_analysis(
            bucket, key, feature_types=DEFAULT_FEATURE_TYPES
        )
# NEW
    try:
        job_id = textract.start_document_text_detection(bucket, key)
3. 
main.py
In process(...) — change the get call:

python

# OLD
    try:
        job = textract.get_document_analysis(job_id)
# NEW
    try:
        job = textract.get_document_text_detection(job_id)
4. 
idp_logic.py
collect_text — replace the body (line-only):

python

def collect_text(structured_text: dict) -> list[str]:
    """Flatten the LINE text of a ``structured_text`` document.

    Gathers ONLY the printed + handwritten LINE text — the single source of truth
    for the pipeline. ``forms``/``tables`` are intentionally excluded: on scanned/
    handwritten documents Textract's structural views are unreliable and diverge
    from the LINE view, which (a) can leave PII un-masked when the same value
    appears as a slightly different literal string in a form/table cell, and (b)
    feeds contradictory/duplicate text to Comprehend + the Agent. The LINE view
    already contains every value on the page, so detection/tokenization stay
    complete. Any ``forms``/``tables`` present are ignored regardless of source.
    Pure and testable.
    """
    parts: list[str] = []
    parts.extend(structured_text.get("printed", []) or [])
    parts.extend(structured_text.get("handwritten", []) or [])
    return [p for p in parts if p]
blocks_to_structured_text — replace with line-only version, and DELETE the _extract_forms, _extract_tables, and _child_word_text helpers:

python

def blocks_to_structured_text(document_id: str, blocks: list[dict]) -> dict:
    """Convert Textract ``Blocks`` into the ``structured_text`` shape (Req 2.2).

    Pure function: printed[]/handwritten[] from LINE ``TextType``. OCR is
    TEXT-ONLY (StartDocumentTextDetection), so no KEY_VALUE_SET/TABLE blocks
    exist; ``forms``/``tables`` are always emitted EMPTY to preserve the
    ``structured_text`` shape/contract. The LINE view is the single source of
    truth for every downstream stage. Works for single- and multi-page (async,
    blocks concatenated across pages) responses.
    """
    printed, handwritten = _extract_lines(blocks)
    return make_structured_text(
        document_id,
        printed=printed,
        handwritten=handwritten,
        forms=[],
        tables=[],
    )
Keep _extract_lines as-is. Delete _child_word_text, _extract_forms, _extract_tables (they're now unused).

5. 
main.py
flatten_tokenized_text — replace the body (line-only). This also fixes the Agent, since 
tools.py
 imports this function:

python

def flatten_tokenized_text(tokenized_text) -> str:
    """Reduce the tokenized-text input to a single string for the prompt.

    Accepts either a plain string or the tokenized-text dict shape produced by
    the tokenizer (``{"sections": {printed, handwritten, forms, tables}}``).
    Only tokens/metadata are present here, never raw PII.

    Uses ONLY the LINE view (``printed`` + ``handwritten``). ``forms``/``tables``
    are intentionally excluded: Textract's structural views are unreliable on
    scanned/handwritten documents and would inject contradictory ``key: value``
    pairs / table rows that confuse classification and field extraction. The LINE
    text already contains every value on the page.

    LIVE CALLER: the IDP Agent reaches this via ``agent/tools.py``
    (``identify_and_extract`` / ``extract_weak_fields`` import this function to
    build the LLM prompt). The deterministic Classifier handler is dormant in the
    current pipeline, so this function governs the AGENT's prompt.
    """
    if isinstance(tokenized_text, str):
        return tokenized_text
    if not isinstance(tokenized_text, dict):
        return str(tokenized_text or "")

    sections = tokenized_text.get("sections", tokenized_text)
    parts: list[str] = []
    parts.extend(str(s) for s in sections.get("printed", []) or [])
    parts.extend(str(s) for s in sections.get("handwritten", []) or [])
    return "\n".join(p for p in parts if p)
6. 
iam.yaml
 — OCR role, Sid: TextractExtract
yaml

              - Sid: TextractExtract
                Effect: Allow
                # Async (multi-page) TEXT-ONLY Textract used by the split OCR:
                # the start stage calls StartDocumentTextDetection, the collect
                # stage calls GetDocumentTextDetection. OCR is text-only (LINE
                # view is the single source of truth), so the analysis actions
                # (StartDocumentAnalysis/GetDocumentAnalysis/AnalyzeDocument) are
                # NOT granted — least privilege.
                Action:
                  - "textract:StartDocumentTextDetection"
                  - "textract:GetDocumentTextDetection"
                Resource: "*"
