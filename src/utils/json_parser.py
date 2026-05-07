"""
PwC Agentic Document Processing — Robust JSON Parser

Handles malformed JSON responses from LLMs with multiple fallback strategies.
"""

import json
import re


def parse_json_robust(text: str) -> dict:
    """Parse JSON from an LLM response, handling common malformations.

    Attempts multiple strategies in order:
    1. Direct JSON parse
    2. Single-quote to double-quote replacement
    3. Brace-bounded extraction
    4. Trailing comma removal

    Args:
        text: Raw text from LLM response.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If all parsing strategies fail.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Replace single quotes
    try:
        fixed = text.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Extract between braces
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            candidate = text[start:end]
            return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Remove trailing commas
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")
