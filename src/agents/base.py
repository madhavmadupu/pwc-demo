"""
PwC Agentic Document Processing — Base Agent

Abstract base class for all pipeline agents with built-in error handling.
"""

from abc import ABC, abstractmethod
from typing import Optional

import streamlit as st
import traceback


class BaseAgent(ABC):
    """Base class for all document processing agents.

    Provides safe execution with error handling and UI feedback.
    Subclasses must implement the `execute` method.
    """

    def __init__(self, name: str) -> None:
        """Initialize the agent with a display name.

        Args:
            name: Human-readable agent name for UI display.
        """
        self.name = name

    def safe_execute(
        self, state: dict, fallback: Optional[dict] = None
    ) -> dict:
        """Execute the agent with error handling.

        Catches all exceptions, displays them in the Streamlit UI,
        logs the error, and applies fallback values if provided.

        Args:
            state: Current pipeline state dictionary.
            fallback: Optional dict of fallback values to apply on failure.

        Returns:
            Updated state dictionary.
        """
        try:
            return self.execute(state)
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            st.warning(f"⚠️ {error_msg}")
            with st.expander(f"🔍 {self.name} Error Details"):
                st.code(traceback.format_exc())
            state.setdefault("pipeline_errors", []).append(error_msg)
            if fallback:
                state.update(fallback)
            return state

    @abstractmethod
    def execute(self, state: dict) -> dict:
        """Execute the agent's core logic.

        Must be implemented by subclasses.

        Args:
            state: Current pipeline state dictionary.

        Returns:
            Updated state dictionary.
        """
        pass
