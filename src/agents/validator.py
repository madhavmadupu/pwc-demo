"""
PwC Agentic Document Processing — Validation Agent

Agent 3: Validates extracted fields against company rules using Gemini.
"""

import json

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import settings
from src.rag.embeddings import get_client
from src.utils.json_parser import parse_json_robust


class ValidationAgent(BaseAgent):
    """Validates extracted document data against company rules and compliance checks."""

    def __init__(self) -> None:
        super().__init__("Validation")

    def execute(self, state: dict) -> dict:
        """Validate extracted fields against applied company rules.

        Args:
            state: Pipeline state with 'extracted_fields', 'doc_type', and 'rules_applied'.

        Returns:
            Updated state with 'validation_result' and 'pipeline_status'.
        """
        extracted = state["extracted_fields"]
        doc_type = state["doc_type"]
        rules = state["rules_applied"]

        if not extracted or "message" in extracted:
            state["validation_result"] = {
                "is_valid": False,
                "score": 0.0,
                "checks": [],
                "issues": ["No data to validate"],
                "warnings": [],
            }
            state["pipeline_status"] = "validated"
            return state

        rules_text = "\n".join([f"  - {rule}" for rule in rules])

        client = get_client()

        prompt = f"""Validate this {doc_type} data against company rules.

Company Rules:
{rules_text}

Extracted Data:
{json.dumps(extracted, indent=2)}

Check: required fields, date validity, amount calculations, completeness.
Return JSON with keys: is_valid (bool), score (0.0-1.0), checks (list), issues (list), warnings (list)."""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature_validation,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("Empty response from API")

        state["validation_result"] = parse_json_robust(response.text)
        state["pipeline_status"] = "validated"
        return state
