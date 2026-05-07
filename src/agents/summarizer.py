"""
PwC Agentic Document Processing — Summarization Agent

Agent 4: Generates executive summaries of processed documents using Gemini.
"""

import json

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import settings
from src.rag.embeddings import get_client
from src.utils.json_parser import parse_json_robust


class SummarizationAgent(BaseAgent):
    """Generates executive summaries from extracted and validated document data."""

    def __init__(self) -> None:
        super().__init__("Summarization")

    def execute(self, state: dict) -> dict:
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

        summary = parse_json_robust(response.text)

        # Ensure required keys exist with safe defaults
        summary.setdefault("title", "Document Summary")
        summary.setdefault("one_liner", "")
        summary.setdefault("key_highlights", [])
        summary.setdefault("action_items", [])
        summary.setdefault("risks", [])
        summary.setdefault("overall_status", "needs_review")

        # Ensure lists are actually lists
        for key in ("key_highlights", "action_items", "risks"):
            val = summary.get(key, [])
            if isinstance(val, str):
                summary[key] = [val]
            elif not isinstance(val, list):
                summary[key] = [str(val)]

        # Normalize overall_status
        status = summary.get("overall_status", "needs_review")
        if isinstance(status, str):
            status_lower = status.lower().strip()
            if status_lower in ("valid", "pass", "passed", "approved", "ok"):
                summary["overall_status"] = "valid"
            else:
                summary["overall_status"] = "needs_review"

        state["summary"] = summary
        state["pipeline_status"] = "completed"
        return state
