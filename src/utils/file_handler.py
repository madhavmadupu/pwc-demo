"""
PwC Agentic Document Processing — File Handling Utilities

PDF text extraction and multimodal image processing using Gemini Vision.
"""

import io

import PyPDF2
import streamlit as st
from google import genai
from google.genai import types

from src.config import settings
from src.rag.embeddings import get_client
from src.utils.json_parser import parse_json_robust


# ============================================================
# PDF Extraction
# ============================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Extracted text from all pages.
    """
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def safe_extract_pdf(uploaded_file) -> str:
    """Safely extract text from a PDF, showing UI warnings on failure.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Extracted text, or empty string on failure.
    """
    try:
        text = extract_text_from_pdf(uploaded_file)
        if not text.strip():
            st.warning("⚠️ PDF appears to be empty or image-based.")
            return ""
        return text
    except Exception as e:
        st.error(f"❌ Failed to extract PDF: {str(e)}")
        return ""


# ============================================================
# Image Processing (Multimodal — Gemini Vision)
# ============================================================
def extract_from_image(image_bytes: bytes, mime_type: str) -> dict:
    """Use Gemini Vision to extract text and describe an image.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the image (e.g., 'image/jpeg').

    Returns:
        Dict with keys: extracted_text, description, has_document,
        detected_type, confidence.
    """
    client = get_client()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    prompt = """Analyze this image carefully. It may be a document (invoice, contract, report, email, letter, form, receipt, ID card, etc.) or a general image.

If it contains text/documents:
1. Extract ALL visible text accurately (OCR)
2. Identify the document type

If it's a general image:
1. Describe what you see in detail
2. Note any text visible in the image

Return JSON with these keys:
- extracted_text: all text found in the image (string, use "" if no text)
- description: detailed description of the image content (string)
- has_document: true if it contains a document/form/invoice/etc, false otherwise
- detected_type: if document, what type (Invoice, Contract, Report, Email, Form, Receipt, ID, Letter, Other)
- confidence: how confident you are in the type detection (0.0-1.0)"""

    response = client.models.generate_content(
        model=settings.model_name,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            temperature=settings.temperature_classification,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise ValueError("Empty response from Vision API")

    return parse_json_robust(response.text)
