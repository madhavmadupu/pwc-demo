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
Return JSON with keys: is_valid (bool), score (0.0-1.0 as a number), checks (list), issues (list), warnings (list).
IMPORTANT: score must be a NUMBER like 0.95, NOT a word."""

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

        validation = parse_json_robust(response.text)

        # Robust score parsing
        score = validation.get("score", 0.0)
        if isinstance(score, str):
            score_lower = score.lower().strip()
            if "%" in score_lower:
                validation["score"] = float(score_lower.replace("%", "").strip()) / 100.0
            elif score_lower in ("high", "pass", "passed", "true", "yes"):
                validation["score"] = 0.95
            elif score_lower in ("medium", "partial"):
                validation["score"] = 0.75
            elif score_lower in ("low", "fail", "failed", "false", "no"):
                validation["score"] = 0.30
            else:
                try:
                    validation["score"] = float(score_lower)
                except ValueError:
                    validation["score"] = 0.50
        elif isinstance(score, (int, float)):
            validation["score"] = float(score)
        else:
            validation["score"] = 0.50

        # Robust is_valid parsing
        is_valid = validation.get("is_valid", False)
        if isinstance(is_valid, str):
            validation["is_valid"] = is_valid.lower().strip() in ("true", "yes", "pass", "passed")
        elif isinstance(is_valid, (int, float)):
            validation["is_valid"] = bool(is_valid)
        else:
            validation["is_valid"] = bool(is_valid)

        # Ensure checks is a list
        checks = validation.get("checks", [])
        if not isinstance(checks, list):
            validation["checks"] = [str(checks)]
        else:
            validation["checks"] = [str(c) if not isinstance(c, (dict, str)) else c for c in checks]

        # Ensure issues and warnings are lists
        for key in ("issues", "warnings"):
            val = validation.get(key, [])
            if not isinstance(val, list):
                validation[key] = [str(val)]

        state["validation_result"] = validation
        state["pipeline_status"] = "validated"
        return state
