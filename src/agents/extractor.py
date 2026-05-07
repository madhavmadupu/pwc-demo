"""
PwC Agentic Document Processing — Extraction Agent

Agent 2: Extracts structured fields from documents using Gemini with RAG context.
"""

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import settings
from src.rag.embeddings import get_client
from src.rag.retriever import get_relevant_rules
from src.rag.rules import COMPANY_RULES
from src.utils.json_parser import parse_json_robust


class ExtractionAgent(BaseAgent):
    """Extracts structured key-value fields from classified documents."""

    def __init__(self) -> None:
        super().__init__("Extraction")

    def execute(self, state: dict) -> dict:
        """Extract fields from the document using RAG-augmented prompts.

        Args:
            state: Pipeline state with 'text' and 'doc_type' keys.

        Returns:
            Updated state with 'extracted_fields', 'rules_applied', and 'pipeline_status'.
        """
        text = state["text"]
        doc_type = state["doc_type"]

        if doc_type == "Unknown":
            state["extracted_fields"] = {"message": "Cannot extract from unknown document type"}
            state["pipeline_status"] = "extracted"
            return state

        # Retrieve relevant rules via RAG
        rules = get_relevant_rules(doc_type, text)
        state["rules_applied"] = COMPANY_RULES.get(doc_type.lower(), [])

        client = get_client()

        prompt = f"""Extract all key fields from this {doc_type} document. Return ONLY JSON.
Include all relevant fields: identifiers, dates, parties, amounts, terms, etc.

{rules}

Document:
---
{text}
---"""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature_extraction,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("Empty response from API")

        state["extracted_fields"] = parse_json_robust(response.text)
        state["pipeline_status"] = "extracted"
        return state
