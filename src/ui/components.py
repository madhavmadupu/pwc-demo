"""
PwC Agentic Document Processing — Reusable UI Components

Streamlit components for rendering pipeline results.
"""

import streamlit as st
import json
from src.rag.rules import ALL_RULES
from src.config import settings


def render_tech_banner() -> None:
    """Render the technology stack metrics banner."""
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🤖 Model", settings.model_name)
    with col2:
        st.metric("👁️ Vision", "Multimodal")
    with col3:
        st.metric("🔗 Orchestration", "LangGraph")
    with col4:
        st.metric("📚 RAG", "Vector Search")
    with col5:
        st.metric("☁️ Deploy", "Cloud Run")
    st.markdown("---")


def render_sidebar() -> None:
    """Render the configuration sidebar."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write(f"**Project:** {settings.project_id}")
        st.write(f"**Region:** {settings.location}")
        st.write("**Pipeline:** 4-Agent Architecture")
        st.write("1. 📄 Classification")
        st.write("2. 🔍 Extraction")
        st.write("3. ✅ Validation")
        st.write("4. 📊 Summary")
        st.write("---")
        st.write("**Input Modes:**")
        st.write("  • 📝 Paste Text")
        st.write("  • 📎 Upload PDF")
        st.write("  • 🖼️ Upload Image")
        st.write("---")
        st.write(f"**RAG Rules:** {len(ALL_RULES)} rules loaded")
        vs_enabled = bool(settings.vector_search_endpoint and settings.deployed_index_id)
        st.write(f"**Vector Search:** {'✅ Enabled' if vs_enabled else '⚠️ In-Memory'}")


def render_classification(result: dict) -> None:
    """Render Agent 1 classification results."""
    st.markdown("### 📄 Agent 1 — Document Classification")
    if result.get("doc_type"):
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Document Type", result["doc_type"])
        with c2:
            conf = result.get("confidence", 0)
            st.metric("Confidence", f"{conf:.0%}")
        if result.get("reasoning"):
            st.info(result["reasoning"])
    else:
        st.warning("Classification was not completed.")


def render_extraction(result: dict) -> None:
    """Render Agent 2 extraction results."""
    st.markdown("### 🔍 Agent 2 — Field Extraction")
    extracted = result.get("extracted_fields", {})
    if extracted and "message" not in extracted:
        st.json(extracted)
    else:
        st.warning("Extraction was not completed.")


def render_validation(result: dict) -> None:
    """Render Agent 3 validation results."""
    st.markdown("### ✅ Agent 3 — Validation & Compliance")
    validation = result.get("validation_result", {})
    if validation:
        c1, c2 = st.columns(2)
        with c1:
            is_valid = validation.get("is_valid", False)
            st.metric("Status", "✅ Passed" if is_valid else "❌ Failed")
        with c2:
            score = validation.get("score", 0)
            if isinstance(score, (int, float)):
                st.metric("Score", f"{score:.0%}")
            else:
                st.metric("Score", str(score))

        if validation.get("checks"):
            for check in validation["checks"]:
                if isinstance(check, dict):
                    icon = "✅" if check.get("status") == "pass" else "⚠️" if check.get("status") == "warning" else "❌"
                    st.write(f"{icon} **{check.get('rule', 'N/A')}** — {check.get('details', '')}")
                else:
                    st.write(f"• {check}")

        if validation.get("issues"):
            for issue in validation["issues"]:
                st.error(f"• {issue}")

        if validation.get("warnings"):
            for warning in validation["warnings"]:
                st.warning(f"• {warning}")
    else:
        st.warning("Validation was not completed.")


def render_summary(result: dict) -> None:
    """Render Agent 4 summary results."""
    st.markdown("### 📊 Agent 4 — Executive Summary")
    summary = result.get("summary", {})
    if summary and summary.get("title"):
        st.write(f"#### {summary['title']}")
        if summary.get("one_liner"):
            st.info(summary["one_liner"])

        if summary.get("key_highlights"):
            st.write("**Key Highlights:**")
            for h in summary["key_highlights"]:
                st.write(f"• {h}")

        if summary.get("action_items"):
            st.write("**Action Items:**")
            for item in summary["action_items"]:
                st.write(f"• {item}")

        if summary.get("risks"):
            st.warning("**Risks:**")
            for risk in summary["risks"]:
                st.write(f"• {risk}")

        if summary.get("overall_status"):
            status_color = "🟢" if summary["overall_status"] == "valid" else "🟡"
            st.write(f"**Overall Status:** {status_color} {summary['overall_status'].replace('_', ' ').title()}")
    else:
        st.warning("Summary was not completed.")


def render_rules_applied(result: dict) -> None:
    """Render RAG rules applied section."""
    if result.get("rules_applied"):
        with st.expander(f"📚 RAG: {len(result['rules_applied'])} Company Rules Applied"):
            for rule in result["rules_applied"]:
                st.write(f"• {rule}")


def render_export(result: dict) -> None:
    """Render export/download section."""
    st.subheader("💾 Export Results")
    export_data = {
        "classification": {
            "type": result.get("doc_type"),
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
        },
        "extracted_fields": result.get("extracted_fields", {}),
        "validation": result.get("validation_result", {}),
        "summary": result.get("summary", {}),
        "rules_applied": result.get("rules_applied", []),
        "pipeline_errors": result.get("pipeline_errors", []),
        "source_type": result.get("source_type", "unknown"),
    }
    st.download_button(
        label="📥 Download JSON Report",
        data=json.dumps(export_data, indent=2, default=str),
        file_name="document_processing_report.json",
        mime="application/json",
    )
