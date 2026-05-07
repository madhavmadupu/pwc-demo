"""
PwC Agentic Document Processing — Main Page Layout

Streamlit page configuration, input handling, pipeline execution, and result display.
"""

import json
import traceback

import streamlit as st

from src.config import settings
from src.models.state import DocumentState
from src.rag.embeddings import get_client
from src.rag.rules import ALL_RULES
from src.utils.file_handler import extract_from_image, safe_extract_pdf
from src.pipeline.orchestrator import build_pipeline
from src.ui.components import (
    render_tech_banner,
    render_sidebar,
    render_classification,
    render_extraction,
    render_validation,
    render_summary,
    render_rules_applied,
    render_export,
)


def render_main_page() -> None:
    """Render the main Streamlit application page."""
    st.set_page_config(
        page_title="PwC Agentic Document Processor",
        page_icon="📄",
        layout="wide",
    )

    # Ensure client is cached
    get_client()

    st.title("📄 PwC Agentic Document Processor")
    st.subheader("AI-Powered Multimodal Document Intelligence")

    render_tech_banner()
    render_sidebar()

    # ---- Input Section ----
    st.header("📝 Input Document")

    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload PDF", "Upload Image"],
        horizontal=True,
    )

    text = ""
    image_description = None

    if input_method == "Paste Text":
        text = st.text_area(
            "Paste your document text here:",
            height=200,
            placeholder="Paste an invoice, contract, report, or email here...",
        )

    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload a PDF document:",
            type=["pdf"],
            help="Supported: Invoice, Contract, Report, Email (PDF format)",
        )
        if uploaded_file:
            text = safe_extract_pdf(uploaded_file)
            if text:
                st.success(f"✅ Extracted {len(text)} characters from PDF")
                with st.expander("Preview extracted text"):
                    st.text(text[:1000])

    elif input_method == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Gemini Vision will extract text and analyze the image",
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Gemini Vision is analyzing the image..."):
                image_bytes = uploaded_image.read()
                mime_map = {
                    "image/jpeg": "image/jpeg",
                    "image/jpg": "image/jpeg",
                    "image/png": "image/png",
                    "image/webp": "image/webp",
                }
                mime_type = mime_map.get(uploaded_image.type, "image/jpeg")

                try:
                    vision_result = extract_from_image(image_bytes, mime_type)

                    st.success("✅ Image analyzed successfully!")

                    if vision_result.get("description"):
                        image_description = vision_result["description"]
                        with st.expander("🖼️ Image Description", expanded=True):
                            st.write(image_description)

                    if vision_result.get("has_document"):
                        st.info(
                            f"📋 **Detected Document Type:** "
                            f"{vision_result.get('detected_type', 'Unknown')} "
                            f"(Confidence: {vision_result.get('confidence', 0):.0%})"
                        )

                    if vision_result.get("extracted_text"):
                        text = vision_result["extracted_text"]
                        st.success(f"✅ Extracted {len(text)} characters from image")
                        with st.expander("📝 Extracted Text", expanded=True):
                            st.text(text[:2000])
                    else:
                        if image_description:
                            text = (
                                f"[Image Analysis — No extractable text found]\n\n"
                                f"Description: {image_description}"
                            )
                            st.info(
                                "ℹ️ No text found in image. Using image description for pipeline analysis."
                            )
                        else:
                            st.warning("⚠️ No text or description extracted from image.")

                except Exception as e:
                    st.error(f"❌ Vision analysis failed: {str(e)}")

    # ---- Process Button ----
    if st.button("🚀 Process Document", type="primary", use_container_width=True):
        if not text.strip():
            st.error("❌ Please provide a document to process.")
            return

        initial_state: DocumentState = {
            "text": text,
            "doc_type": "",
            "confidence": 0.0,
            "reasoning": "",
            "extracted_fields": {},
            "validation_result": {},
            "summary": {},
            "rules_applied": [],
            "pipeline_status": "starting",
            "pipeline_errors": [],
            "source_type": input_method.lower().replace(" ", "_"),
        }

        pipeline = build_pipeline()

        st.info("🔄 Starting 4-Agent Pipeline...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("📄 Agent 1: Classifying document...")
            progress_bar.progress(25)
            # The pipeline handles safe_execute internally via BaseAgent
            # but we run agents individually for progress tracking
            from src.agents.classifier import ClassificationAgent
            from src.agents.extractor import ExtractionAgent
            from src.agents.validator import ValidationAgent
            from src.agents.summarizer import SummarizationAgent

            classifier = ClassificationAgent()
            result = classifier.safe_execute(
                initial_state,
                fallback={
                    "doc_type": "Unknown",
                    "confidence": 0.0,
                    "reasoning": "Classification failed",
                },
            )

            status_text.text("🔍 Agent 2: Extracting fields...")
            progress_bar.progress(50)
            extractor = ExtractionAgent()
            result = extractor.safe_execute(
                result,
                fallback={"extracted_fields": {"message": "Extraction failed"}},
            )

            status_text.text("✅ Agent 3: Validating data...")
            progress_bar.progress(75)
            validator = ValidationAgent()
            result = validator.safe_execute(
                result,
                fallback={
                    "validation_result": {
                        "is_valid": False,
                        "score": 0.0,
                        "checks": [],
                        "issues": ["Validation failed"],
                        "warnings": [],
                    }
                },
            )

            status_text.text("📊 Agent 4: Generating summary...")
            progress_bar.progress(90)
            summarizer = SummarizationAgent()
            result = summarizer.safe_execute(
                result,
                fallback={
                    "summary": {
                        "title": "Summary unavailable",
                        "one_liner": "Summarization failed",
                        "key_highlights": [],
                        "action_items": [],
                    }
                },
            )

            progress_bar.progress(100)
            status_text.text("✅ Pipeline complete!")

        except Exception as e:
            st.error(f"❌ Unexpected pipeline error: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            result = initial_state

        # ---- Display Results ----
        st.markdown("---")

        agents_done = sum([
            1 if result.get("doc_type") else 0,
            1 if result.get("extracted_fields") and "message" not in result.get("extracted_fields", {}) else 0,
            1 if result.get("validation_result") else 0,
            1 if result.get("summary", {}).get("title") else 0,
        ])
        st.success(f"✅ **Pipeline Complete** — {agents_done}/4 agents successful")

        if result.get("pipeline_errors"):
            for error in result["pipeline_errors"]:
                st.warning(f"⚠️ {error}")

        render_classification(result)
        render_extraction(result)
        render_validation(result)
        render_summary(result)
        render_rules_applied(result)
        render_export(result)
