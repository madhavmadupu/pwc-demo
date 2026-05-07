"""Tests for agent implementations."""

import pytest
from unittest.mock import MagicMock, patch


class TestBaseAgent:
    """Tests for the BaseAgent class."""

    def test_base_agent_cannot_be_instantiated(self):
        """BaseAgent is abstract and cannot be instantiated directly."""
        from src.agents.base import BaseAgent

        with pytest.raises(TypeError):
            BaseAgent("test")

    def test_safe_execute_calls_execute(self):
        """safe_execute should call the execute method."""
        from src.agents.classifier import ClassificationAgent

        agent = ClassificationAgent()
        mock_state = {"text": "test", "doc_type": "", "pipeline_errors": []}

        with patch.object(agent, "execute", return_value=mock_state) as mock_exec:
            result = agent.safe_execute(mock_state)
            mock_exec.assert_called_once()
            assert result == mock_state

    def test_safe_execute_handles_errors(self):
        """safe_execute should catch exceptions and apply fallback."""
        from src.agents.classifier import ClassificationAgent

        agent = ClassificationAgent()
        mock_state = {"text": "test", "pipeline_errors": []}
        fallback = {"doc_type": "Unknown", "confidence": 0.0, "reasoning": "Failed"}

        with patch.object(agent, "execute", side_effect=RuntimeError("API error")):
            result = agent.safe_execute(mock_state, fallback=fallback)
            assert result["doc_type"] == "Unknown"
            assert any("Classification failed" in e for e in result["pipeline_errors"])


class TestClassificationAgent:
    """Tests for the ClassificationAgent."""

    def test_agent_name(self):
        from src.agents.classifier import ClassificationAgent

        agent = ClassificationAgent()
        assert agent.name == "Classification"


class TestExtractionAgent:
    """Tests for the ExtractionAgent."""

    def test_agent_name(self):
        from src.agents.extractor import ExtractionAgent

        agent = ExtractionAgent()
        assert agent.name == "Extraction"


class TestValidationAgent:
    """Tests for the ValidationAgent."""

    def test_agent_name(self):
        from src.agents.validator import ValidationAgent

        agent = ValidationAgent()
        assert agent.name == "Validation"


class TestSummarizationAgent:
    """Tests for the SummarizationAgent."""

    def test_agent_name(self):
        from src.agents.summarizer import SummarizationAgent

        agent = SummarizationAgent()
        assert agent.name == "Summarization"
