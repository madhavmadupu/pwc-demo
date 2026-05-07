"""Shared test fixtures for the PwC Agentic Document Processing test suite."""

import pytest


@pytest.fixture
def sample_invoice_text() -> str:
    """Sample invoice document text for testing."""
    return """
    INVOICE #INV-2024-001
    Date: January 15, 2024
    From: TechCorp Solutions Pvt Ltd
    To: PwC India
    
    Items:
    1. Cloud Infrastructure Services - ₹50,000
    2. AI Model Training - ₹75,000
    3. Data Processing - ₹25,000
    
    Subtotal: ₹1,50,000
    Tax (18%): ₹27,000
    Total: ₹1,77,000
    
    Payment Terms: Net 30 days
    Bank: HDFC Bank, Account: 1234567890
    """


@pytest.fixture
def sample_contract_text() -> str:
    """Sample contract document text for testing."""
    return """
    SERVICE AGREEMENT
    
    Effective Date: March 1, 2024
    Parties: TechCorp (Service Provider) and PwC (Client)
    
    Term: 12 months
    Compensation: ₹5,00,000 per month
    
    Termination: Either party may terminate with 30 days written notice.
    Confidentiality: Both parties agree to maintain confidentiality.
    IP Rights: All work product remains property of the client.
    """


@pytest.fixture
def sample_initial_state() -> dict:
    """Initial pipeline state for testing."""
    return {
        "text": "Test document text",
        "doc_type": "",
        "confidence": 0.0,
        "reasoning": "",
        "extracted_fields": {},
        "validation_result": {},
        "summary": {},
        "rules_applied": [],
        "pipeline_status": "starting",
        "pipeline_errors": [],
        "source_type": "paste_text",
    }
