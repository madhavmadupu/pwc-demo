"""
PwC Agentic Document Processing — Configuration

Centralized configuration using dataclasses for type safety and IDE support.
All environment variables and constants are defined here.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    # Google Cloud
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "pwc-agentic-demo")
    location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Vertex AI / Gemini Models
    model_name: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"

    # RAG Configuration
    max_retrieved_rules: int = 8

    # Agent Temperatures
    temperature_classification: float = 0.1
    temperature_extraction: float = 0.1
    temperature_validation: float = 0.1
    temperature_summary: float = 0.3

    # Vector Search (optional — falls back to in-memory rules)
    vector_search_endpoint: str = os.getenv("VECTOR_SEARCH_ENDPOINT", "")
    deployed_index_id: str = os.getenv("DEPLOYED_INDEX_ID", "")


# Singleton settings instance
settings = Settings()
