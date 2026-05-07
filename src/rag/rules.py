"""
PwC Agentic Document Processing — RAG Rules

Company rules and standard operating procedures (SOPs) for document
validation. These rules are used by the RAG retrieval system to provide
context-relevant validation criteria to the agents.
"""

from typing import Dict, List


# ============================================================
# Company Rules & SOPs — Organized by document type
# ============================================================
COMPANY_RULES: Dict[str, List[str]] = {
    "invoice": [
        "All invoices must have a valid invoice number",
        "Invoice date must not be in the future",
        "Tax calculation must match: subtotal × tax rate = tax amount",
        "Line items must sum to subtotal",
        "Payment terms must be specified (Net 15-90 days)",
        "Bank details are required for payments above ₹50,000",
        "All amounts must be in positive currency",
        "Vendor name and address are mandatory",
    ],
    "contract": [
        "Both parties must be clearly identified",
        "Effective date and term duration must be specified",
        "Compensation terms must be clearly stated",
        "A termination clause is mandatory",
        "Confidentiality clause is required",
        "IP rights must be addressed",
        "Dispute resolution mechanism must be specified",
        "Force majeure clause is recommended",
    ],
    "report": [
        "Report must have a clear title and period",
        "Executive summary must be present (>50 chars)",
        "At least 3 key metrics with values required",
        "Trend indicators (up/down/stable) for each metric",
        "Challenges must be documented",
        "Actionable recommendations required",
        "Data sources should be cited",
        "Charts/graphs recommended for visual reports",
    ],
    "email": [
        "Must have valid sender and recipient",
        "Subject line must be present and descriptive",
        "Date must be parseable",
        "Body summary must be coherent (>20 chars)",
        "Action items should be clearly identified",
        "Sentiment analysis required",
        "Urgency level must be classified",
        "Attachments should be noted if present",
    ],
}


def flatten_rules() -> List[dict]:
    """Flatten all company rules into a list of dicts for vector search indexing.

    Returns:
        List of dicts with 'text', 'doc_type', and 'id' keys.
    """
    all_rules: List[dict] = []
    for doc_type, rules in COMPANY_RULES.items():
        for i, rule in enumerate(rules):
            all_rules.append({
                "text": rule,
                "doc_type": doc_type,
                "id": f"{doc_type}_{i}",
            })
    return all_rules


# Pre-flattened rules for vector search
ALL_RULES: List[dict] = flatten_rules()

# ID-to-text mapping for vector search results lookup
RULE_ID_MAP: dict[str, str] = {r["id"]: r["text"] for r in ALL_RULES}
