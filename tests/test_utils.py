"""Tests for utility functions."""

import json
import pytest

from src.utils.json_parser import parse_json_robust


class TestParseJsonRobust:
    """Tests for the robust JSON parser."""

    def test_valid_json(self):
        """Should parse valid JSON directly."""
        result = parse_json_robust('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_code_fence(self):
        """Should strip markdown code fences."""
        result = parse_json_robust('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_single_quotes(self):
        """Should handle single-quote replacement."""
        result = parse_json_robust("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        """Should extract JSON from surrounding text."""
        result = parse_json_robust('Here is the result: {"key": "value"} done.')
        assert result == {"key": "value"}

    def test_json_with_trailing_comma(self):
        """Should handle trailing commas."""
        result = parse_json_robust('{"key": "value",}')
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        """Should raise ValueError for unparseable input."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_robust("not json at all")

    def test_nested_json(self):
        """Should handle nested objects."""
        data = {"outer": {"inner": [1, 2, 3]}}
        result = parse_json_robust(json.dumps(data))
        assert result == data
