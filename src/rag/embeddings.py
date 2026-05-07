"""
PwC Agentic Document Processing — Embedding Computation

Handles embedding generation using Gemini Embeddings for the RAG pipeline.
"""

import streamlit as st
from google import genai

from src.config import settings


@st.cache_resource
def get_client() -> genai.Client:
    """Create a cached Vertex AI Gemini client.

    Returns:
        A genai.Client configured for Vertex AI.
    """
    return genai.Client(
        vertexai=True,
        project=settings.project_id,
        location=settings.location,
    )


def compute_embedding(client: genai.Client, text: str) -> list[float]:
    """Compute an embedding vector for the given text.

    Args:
        client: The Gemini client instance.
        text: Text to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=text,
    )
    return response.embeddings[0].values
