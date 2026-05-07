"""
PwC Agentic Document Processing — Pipeline Orchestrator

LangGraph StateGraph that chains the four agents into a sequential pipeline.
"""

from langgraph.graph import StateGraph, END

from src.models.state import DocumentState
from src.agents.classifier import ClassificationAgent
from src.agents.extractor import ExtractionAgent
from src.agents.validator import ValidationAgent
from src.agents.summarizer import SummarizationAgent


def build_pipeline():
    """Build and compile the LangGraph processing pipeline.

    The pipeline consists of four sequential stages:
    1. Classification → 2. Extraction → 3. Validation → 4. Summarization

    Returns:
        Compiled LangGraph runnable.
    """
    # Instantiate agents
    classifier = ClassificationAgent()
    extractor = ExtractionAgent()
    validator = ValidationAgent()
    summarizer = SummarizationAgent()

    # Build the graph
    workflow = StateGraph(DocumentState)

    workflow.add_node("classify", classifier.safe_execute)
    workflow.add_node("extract", extractor.safe_execute)
    workflow.add_node("validate", validator.safe_execute)
    workflow.add_node("summarize", summarizer.safe_execute)

    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()
