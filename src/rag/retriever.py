"""
PwC Agentic Document Processing — RAG Retriever

Retrieves relevant company rules using Vertex AI Vector Search or
an in-memory fallback with cosine similarity.
"""

import math
import os
from typing import List

import streamlit as st

from src.config import settings
from src.rag.rules import COMPANY_RULES, RULE_ID_MAP
from src.rag.embeddings import get_client, compute_embedding


# ============================================================
# Cosine Similarity (fallback for in-memory vector search)
# ============================================================
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ============================================================
# Rule Embeddings Cache
# ============================================================
@st.cache_data
def get_rule_embeddings() -> dict:
    """Pre-compute embeddings for all company rules.

    Returns:
        Dict mapping rule text to its embedding vector.
    """
    client = get_client()
    embeddings: dict = {}
    for rule_text in COMPANY_RULES.get("invoice", []) + \
                     COMPANY_RULES.get("contract", []) + \
                     COMPANY_RULES.get("report", []) + \
                     COMPANY_RULES.get("email", []):
        if rule_text not in embeddings:
            embeddings[rule_text] = compute_embedding(client, rule_text)
    return embeddings


# ============================================================
# Main RAG Retrieval
# ============================================================
def get_relevant_rules(doc_type: str, text: str) -> str:
    """Retrieve relevant rules for a document using RAG.

    Tries Vertex AI Vector Search first, falls back to in-memory
    cosine similarity, then to simple rule lookup.

    Args:
        doc_type: The classified document type.
        text: The document text for embedding comparison.

    Returns:
        Formatted string of relevant rules.
    """
    # Attempt 1: Vertex AI Vector Search (production)
    try:
        from google.cloud import aiplatform

        aiplatform.init(project=settings.project_id, location=settings.location)
        client = get_client()
        query_embedding = compute_embedding(
            client,
            f"{doc_type} document validation rules: {text[:500]}",
        )

        endpoint_name = settings.vector_search_endpoint
        index_id = settings.deployed_index_id

        if endpoint_name and index_id:
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_name,
            )
            response = index_endpoint.find_neighbors(
                deployed_index_id=index_id,
                queries=[query_embedding],
                num_neighbors=10,
            )
            rules = [
                RULE_ID_MAP.get(neighbor.id, neighbor.id)
                for neighbor in response[0]
                if neighbor.id in RULE_ID_MAP
            ]
            rules_text = "\n".join([f"  - {rule}" for rule in rules])
            return f"Relevant Rules (from Vector Search):\n{rules_text}"
    except Exception:
        pass

    # Attempt 2: In-memory cosine similarity (development)
    doc_type_lower = doc_type.lower()
    if doc_type_lower not in COMPANY_RULES:
        return "No specific rules found for this document type."

    try:
        client = get_client()
        query_embedding = compute_embedding(
            client,
            f"{doc_type} document validation rules: {text[:500]}",
        )
        rule_embeddings = get_rule_embeddings()

        scored_rules: list[tuple[float, str]] = []
        for rule_text, rule_emb in rule_embeddings.items():
            score = cosine_similarity(query_embedding, rule_emb)
            scored_rules.append((score, rule_text))

        scored_rules.sort(key=lambda x: x[0], reverse=True)
        top_rules = [rule for _, rule in scored_rules[:settings.max_retrieved_rules]]
        rules_text = "\n".join([f"  - {rule}" for rule in top_rules])
        return f"Relevant Rules (Cosine Similarity):\n{rules_text}"
    except Exception:
        pass

    # Attempt 3: Simple static rule lookup (offline / no API key)
    rules = COMPANY_RULES.get(doc_type_lower, [])
    rules_text = "\n".join([f"  {i+1}. {rule}" for i, rule in enumerate(rules)])
    return f"Company Rules & SOPs for {doc_type}:\n{rules_text}"
