"""
PwC Agentic Document Processing — Data Models

TypedDict definitions for the LangGraph pipeline state.
"""

from typing import TypedDict


class DocumentState(TypedDict, total=False):
    """State dictionary passed between agents in the LangGraph pipeline.

    Attributes:
        text: Raw document text (extracted from PDF, image, or pasted).
        doc_type: Classified document type (Invoice, Contract, Report, Email, Unknown).
        confidence: Classification confidence score (0.0–1.0).
        reasoning: Classification reasoning from the model.
        extracted_fields: Key-value pairs extracted from the document.
        validation_result: Validation outcome with checks, issues, and warnings.
        summary: Executive summary with highlights, action items, and risks.
        rules_applied: List of company rules used during extraction/validation.
        pipeline_status: Current pipeline stage status.
        pipeline_errors: List of error messages encountered during processing.
        source_type: Input source type (paste_text, upload_pdf, upload_image).
    """

    text: str
    doc_type: str
    confidence: float
    reasoning: str
    extracted_fields: dict
    validation_result: dict
    summary: dict
    rules_applied: list
    pipeline_status: str
    pipeline_errors: list
    source_type: str
