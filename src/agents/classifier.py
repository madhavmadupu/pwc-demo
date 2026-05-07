"""
PwC Agentic Document Processing — Classification Agent

Agent 1: Classifies documents into categories using Gemini.
"""

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import settings
from src.rag.embeddings import get_client
from src.utils.json_parser import parse_json_robust


class ClassificationAgent(BaseAgent):
    """Classifies a document into a category (Invoice, Contract, Report, Email, Unknown)."""

    def __init__(self) -> None:
        super().__init__("Classification")

    def execute(self, state: dict) -> dict:
        """Classify the document text.

        Args:
            state: Pipeline state with 'text' key.

        Returns:
            Updated state with 'doc_type', 'confidence', 'reasoning', and 'pipeline_status'.
        """
        text = state["text"]
        if not text.strip():
            raise ValueError("Empty document text provided")

        client = get_client()

        prompt = f"""You are a document classification expert.
Classify this document into exactly ONE category: Invoice, Contract, Report, Email, Unknown.
Return JSON with keys: type, confidence, reasoning.

Document:
---
{text}
---"""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature_classification,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("Empty response from API")

        result = parse_json_robust(response.text)
        required_keys = ["type", "confidence", "reasoning"]
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing key in response: {key}")

        state["doc_type"] = result["type"]
        state["confidence"] = float(result["confidence"])
        state["reasoning"] = result["reasoning"]
        state["pipeline_status"] = "classified"
        return state
