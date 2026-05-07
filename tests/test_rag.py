"""Tests for RAG components."""

import math
import pytest

from src.rag.rules import COMPANY_RULES, ALL_RULES, flatten_rules
from src.rag.retriever import cosine_similarity


class TestRules:
    """Tests for company rules definitions."""

    def test_all_document_types_present(self):
        """All expected document types should have rules."""
        expected_types = {"invoice", "contract", "report", "email"}
        assert set(COMPANY_RULES.keys()) == expected_types

    def test_rules_are_non_empty(self):
        """Each document type should have at least one rule."""
        for doc_type, rules in COMPANY_RULES.items():
            assert len(rules) > 0, f"{doc_type} has no rules"

    def test_flatten_rules_structure(self):
        """Flattened rules should have text, doc_type, and id keys."""
        flat = flatten_rules()
        for rule in flat:
            assert "text" in rule
            assert "doc_type" in rule
            assert "id" in rule

    def test_flatten_rules_count(self):
        """Flattened rules count should match sum of all rule lists."""
        expected = sum(len(r) for r in COMPANY_RULES.values())
        assert len(ALL_RULES) == expected

    def test_flatten_rules_ids_unique(self):
        """Each flattened rule should have a unique ID."""
        flat = flatten_rules()
        ids = [r["id"] for r in flat]
        assert len(ids) == len(set(ids))


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        v = [1.0, 2.0, 3.0]
        assert math.isclose(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(cosine_similarity(a, b), -1.0)

    def test_zero_vector(self):
        """Zero vector should return 0.0 (safe division)."""
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_empty_vectors(self):
        """Empty vectors should return 0.0."""
        assert cosine_similarity([], []) == 0.0
