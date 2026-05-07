"""
PwC Agentic Document Processing — Summarization Agent

Agent 4: Generates executive summaries from extracted and validated data.
"""

import json

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import settings
from src.rag.embeddings import get_client
from src.utils.json_parser import parse_json_robust


class SummarizationAgent(BaseAgent):
    """Generates executive summaries with highlights, action items, and risks."""

    def __init__(self) -> None:
        super().__init__("Summarization")

    def execute(self, state: dict) -> dict:
        """Generate an executive summary from extracted and validated data.

        Args:
            state: Pipeline state with 'extracted_fields', 'validation_result', and 'doc_type'.

        Returns:
            Updated state with 'summary' and 'pipeline_status'.
        """
        extracted = state["extracted_fields"]
        validation = state["validation_result"]
        doc_type = state["doc_type"]

        client = get_client()

        prompt = f"""Generate a concise executive summary of this {doc_type}.

Extracted Data:
{json.dumps(extracted, indent=2)}

Validation Result:
{json.dumps(validation, indent=2)}

Return JSON with keys:
- title: string
- one_liner: one sentence summary
- key_highlights: list of 3 highlights
- action_items: list of actions
- risks: list of risks (if any)
- overall_status: "valid" or "needs_review" based on validation"""

        response = client.models.generate_content(
            model=settings.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=settings.temperature_summary,
                response_mime_type="application/json",
            ),
        )

        if not response.text:
            raise ValueError("Empty response from API")

        state["summary"] = parse_json_robust(response.text)
        state["pipeline_status"] = "completed"
        return state
