"""
PwC Agentic Document Processing — Streamlit UI
Tech: Gemini 2.5 Flash + LangGraph + Vertex AI Vector Search RAG + Streamlit
Deploy: Cloud Run
"""

import streamlit as st
import json
import os
from google import genai
from google.genai import types
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import PyPDF2
import io
import traceback

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="PwC Agentic Document Processor",
    page_icon="📄",
    layout="wide"
)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "pwc-agentic-demo")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# ============================================================
# VERTEX AI CLIENT
# ============================================================
@st.cache_resource
def get_client():
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

client = get_client()

# ============================================================
# RAG — Company Rules & SOPs (Vector Search)
# ============================================================
COMPANY_RULES = {
    "invoice": [
        "All invoices must have a valid invoice number",
        "Invoice date must not be in the future",
        "Tax calculation must match: subtotal × tax rate = tax amount",
        "Line items must sum to subtotal",
        "Payment terms must be specified (Net 15-90 days)",
        "Bank details are required for payments above ₹50,000",
        "All amounts must be in positive currency",
        "Vendor name and address are mandatory"
    ],
    "contract": [
        "Both parties must be clearly identified",
        "Effective date and term duration must be specified",
        "Compensation terms must be clearly stated",
        "A termination clause is mandatory",
        "Confidentiality clause is required",
        "IP rights must be addressed",
        "Dispute resolution mechanism must be specified",
        "Force majeure clause is recommended"
    ],
    "report": [
        "Report must have a clear title and period",
        "Executive summary must be present (>50 chars)",
        "At least 3 key metrics with values required",
        "Trend indicators (up/down/stable) for each metric",
        "Challenges must be documented",
        "Actionable recommendations required",
        "Data sources should be cited",
        "Charts/graphs recommended for visual reports"
    ],
    "email": [
        "Must have valid sender and recipient",
        "Subject line must be present and descriptive",
        "Date must be parseable",
        "Body summary must be coherent (>20 chars)",
        "Action items should be clearly identified",
        "Sentiment analysis required",
        "Urgency level must be classified",
        "Attachments should be noted if present"
    ]
}

# Flatten all rules for Vector Search
ALL_RULES = []
for doc_type, rules in COMPANY_RULES.items():
    for rule in rules:
        ALL_RULES.append({
            "text": rule,
            "doc_type": doc_type,
            "id": f"{doc_type}_{len(ALL_RULES)}"
        })

def get_relevant_rules(doc_type: str, text: str) -> str:
    """
    RAG: Retrieve relevant rules.
    Uses Vertex AI Embeddings + Vector Search when available,
    falls back to in-memory rules.
    """
    try:
        # Try Vector Search first (when deployed on Cloud Run with index)
        from google.cloud import aiplatform
        
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Get embedding for the query
        embed_response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=f"{doc_type} document validation rules: {text[:500]}"
        )
        query_embedding = embed_response.embeddings[0].values
        
        # Query Vector Search index (if available)
        index_endpoint_name = os.environ.get("VECTOR_SEARCH_ENDPOINT")
        deployed_index_id = os.environ.get("DEPLOYED_INDEX_ID")
        
        if index_endpoint_name and deployed_index_id:
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=index_endpoint_name
            )
            
            response = index_endpoint.find_neighbors(
                deployed_index_id=deployed_index_id,
                queries=[query_embedding],
                num_neighbors=10
            )
            
            rules = [neighbor.id for neighbor in response[0]]
            rules_text = "\n".join([f"  - {rule}" for rule in rules])
            return f"Relevant Rules (from Vector Search):\n{rules_text}"
        
    except Exception as e:
        # Fall back to in-memory rules
        pass
    
    # Fallback: Use in-memory rules
    doc_type_lower = doc_type.lower()
    if doc_type_lower not in COMPANY_RULES:
        return "No specific rules found for this document type."
    rules = COMPANY_RULES[doc_type_lower]
    rules_text = "\n".join([f"  {i+1}. {rule}" for i, rule in enumerate(rules)])
    return f"Company Rules & SOPs for {doc_type}:\n{rules_text}"

# ============================================================
# ERROR HANDLER
# ============================================================
def safe_agent(name: str, func, state: dict, fallback_value=None) -> dict:
    """Run an agent safely with error handling."""
    try:
        return func(state)
    except Exception as e:
        error_msg = f"{name} failed: {str(e)}"
        st.warning(f"⚠️ {error_msg}")
        with st.expander(f"🔍 {name} Error Details"):
            st.code(traceback.format_exc())
        state["pipeline_errors"].append(error_msg)
        if fallback_value is not None:
            state.update(fallback_value)
        return state

# ============================================================
# AGENT STATE (LangGraph)
# ============================================================
class DocumentState(TypedDict):
    text: str
    doc_type: str
    confidence: float
    reasoning: str
    extracted_fields: dict
    validation_result: dict
    summary: dict
    rules_applied: list
    pipeline_status: str
    pipeline_errors: list

# ============================================================
# AGENT 1: Classification
# ============================================================
def classify_agent(state: DocumentState) -> DocumentState:
    text = state["text"]
    if not text.strip():
        raise ValueError("Empty document text provided")
    
    prompt = f"""You are a document classification expert.
Classify this document into exactly ONE category: Invoice, Contract, Report, Email, Unknown.
Return JSON with keys: type, confidence, reasoning.

Document:
---
{text}
---"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    if not response.text:
        raise ValueError("Empty response from API")
    
    result = json.loads(response.text)
    required_keys = ["type", "confidence", "reasoning"]
    for key in required_keys:
        if key not in result:
            raise ValueError(f"Missing key in response: {key}")
    
    state["doc_type"] = result["type"]
    state["confidence"] = float(result["confidence"])
    state["reasoning"] = result["reasoning"]
    state["pipeline_status"] = "classified"
    return state

# ============================================================
# AGENT 2: Extraction (with RAG)
# ============================================================
def extract_agent(state: DocumentState) -> DocumentState:
    text = state["text"]
    doc_type = state["doc_type"]
    
    if doc_type == "Unknown":
        state["extracted_fields"] = {"message": "Cannot extract from unknown document type"}
        state["pipeline_status"] = "extracted"
        return state
    
    # RAG: Get relevant rules
    rules = get_relevant_rules(doc_type, text)
    state["rules_applied"] = COMPANY_RULES.get(doc_type.lower(), [])
    
    prompt = f"""Extract all key fields from this {doc_type} document. Return ONLY JSON.
Include all relevant fields: identifiers, dates, parties, amounts, terms, etc.

{rules}

Document:
---
{text}
---"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    if not response.text:
        raise ValueError("Empty response from API")
    
    state["extracted_fields"] = json.loads(response.text)
    state["pipeline_status"] = "extracted"
    return state

# ============================================================
# AGENT 3: Validation (with RAG)
# ============================================================
def validate_agent(state: DocumentState) -> DocumentState:
    extracted = state["extracted_fields"]
    doc_type = state["doc_type"]
    rules = state["rules_applied"]
    
    if not extracted or "message" in extracted:
        state["validation_result"] = {
            "is_valid": False,
            "score": 0.0,
            "checks": [],
            "issues": ["No data to validate"],
            "warnings": []
        }
        state["pipeline_status"] = "validated"
        return state
    
    rules_text = "\n".join([f"  - {rule}" for rule in rules])
    
    prompt = f"""Validate this {doc_type} data against company rules.

Company Rules:
{rules_text}

Extracted Data:
{json.dumps(extracted, indent=2)}

Check: required fields, date validity, amount calculations, completeness.
Return JSON with keys: is_valid (bool), score (0.0-1.0), checks (list), issues (list), warnings (list)."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    
    if not response.text:
        raise ValueError("Empty response from API")
    
    state["validation_result"] = json.loads(response.text)
    state["pipeline_status"] = "validated"
    return state

# ============================================================
# AGENT 4: Summary
# ============================================================
def summarize_agent(state: DocumentState) -> DocumentState:
    extracted = state["extracted_fields"]
    validation = state["validation_result"]
    doc_type = state["doc_type"]
    
    prompt = f"""Generate a concise executive summary of this {doc_type}.

Extracted Data:
{json.dumps(extracted, indent=2)}

Validation Result:
{json.dumps(validation, indent=2)}

Return JSON with keys:
- title: string
- one_liner: one sentence summary
- key_highlights: list of 3 highlights
- action_items: list of actions
- risks: list of risks (if any)
- overall_status: "valid" or "needs_review" based on validation"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )
    )
    
    if not response.text:
        raise ValueError("Empty response from API")
    
    state["summary"] = json.loads(response.text)
    state["pipeline_status"] = "completed"
    return state

# ============================================================
# LANGGRAPH PIPELINE
# ============================================================
def build_pipeline():
    workflow = StateGraph(DocumentState)
    
    workflow.add_node("classify", classify_agent)
    workflow.add_node("extract", extract_agent)
    workflow.add_node("validate", validate_agent)
    workflow.add_node("summarize", summarize_agent)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "extract")
    workflow.add_edge("extract", "validate")
    workflow.add_edge("validate", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow.compile()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_text_from_pdf(uploaded_file) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def safe_extract_pdf(uploaded_file) -> str:
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
# STREAMLIT UI
# ============================================================
def main():
    st.title("📄 PwC Agentic Document Processor")
    st.subheader("AI-Powered Document Classification, Extraction, Validation & Summarization")
    
    # Tech stack banner
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Model", "Gemini 2.5 Flash")
    with col2:
        st.metric("🔗 Orchestration", "LangGraph")
    with col3:
        st.metric("📚 RAG", "Vector Search")
    with col4:
        st.metric("☁️ Deploy", "Cloud Run")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write(f"**Project:** {PROJECT_ID}")
        st.write(f"**Region:** {LOCATION}")
        st.write("**Pipeline:** 4-Agent Architecture")
        st.write("1. 📄 Classification")
        st.write("2. 🔍 Extraction")
        st.write("3. ✅ Validation")
        st.write("4. 📊 Summary")
        st.write("---")
        st.write(f"**RAG Rules:** {len(ALL_RULES)} rules loaded")
        st.write(f"**Vector Search:** {'✅ Enabled' if os.environ.get('VECTOR_SEARCH_ENDPOINT') else '⚠️ In-Memory'}")
    
    # Input section
    st.header("📝 Input Document")
    
    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload PDF"],
        horizontal=True
    )
    
    text = ""
    
    if input_method == "Paste Text":
        text = st.text_area(
            "Paste your document text here:",
            height=200,
            placeholder="Paste an invoice, contract, report, or email here..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a PDF document:",
            type=["pdf"],
            help="Supported: Invoice, Contract, Report, Email (PDF format)"
        )
        if uploaded_file:
            text = safe_extract_pdf(uploaded_file)
            if text:
                st.success(f"✅ Extracted {len(text)} characters from PDF")
                with st.expander("Preview extracted text"):
                    st.text(text[:1000])
    
    # Process button
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
            "pipeline_errors": []
        }
        
        pipeline = build_pipeline()
        
        st.info("🔄 Starting 4-Agent Pipeline...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("📄 Agent 1: Classifying document...")
            progress_bar.progress(25)
            result = safe_agent("Classification", classify_agent, initial_state,
                              fallback_value={"doc_type": "Unknown", "confidence": 0.0, "reasoning": "Classification failed"})
            
            status_text.text("🔍 Agent 2: Extracting fields...")
            progress_bar.progress(50)
            result = safe_agent("Extraction", extract_agent, result,
                              fallback_value={"extracted_fields": {"message": "Extraction failed"}})
            
            status_text.text("✅ Agent 3: Validating data...")
            progress_bar.progress(75)
            result = safe_agent("Validation", validate_agent, result,
                              fallback_value={"validation_result": {"is_valid": False, "score": 0.0, "checks": [], "issues": ["Validation failed"], "warnings": []}})
            
            status_text.text("📊 Agent 4: Generating summary...")
            progress_bar.progress(90)
            result = safe_agent("Summarization", summarize_agent, result,
                              fallback_value={"summary": {"title": "Summary unavailable", "one_liner": "Summarization failed", "key_highlights": [], "action_items": []}})
            
            progress_bar.progress(100)
            status_text.text("✅ Pipeline complete!")
            
        except Exception as e:
            st.error(f"❌ Unexpected pipeline error: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
            result = initial_state
        
        # Display results
        st.markdown("---")
        st.header("📊 Pipeline Results")
        
        if result.get("pipeline_errors"):
            st.warning(f"⚠️ Pipeline completed with {len(result['pipeline_errors'])} warning(s):")
            for error in result["pipeline_errors"]:
                st.write(f"  • {error}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("✅ Classified") if result["doc_type"] else st.error("❌ Failed")
        with col2:
            st.success("✅ Extracted") if result["extracted_fields"] and "message" not in result["extracted_fields"] else st.warning("⚠️ Incomplete")
        with col3:
            if result["validation_result"].get("is_valid"):
                st.success("✅ Passed")
            elif result["validation_result"]:
                st.error("❌ Failed")
            else:
                st.warning("⚠️ Skipped")
        with col4:
            st.success("✅ Summarized") if result["summary"].get("title") else st.warning("⚠️ Incomplete")
        
        if result["doc_type"]:
            st.subheader("📄 Agent 1: Classification")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Document Type", result["doc_type"])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.0%}")
            with col3:
                st.metric("Status", result["pipeline_status"].title())
            if result["reasoning"]:
                st.info(f"**Reasoning:** {result['reasoning']}")
        
        if result["extracted_fields"] and "message" not in result["extracted_fields"]:
            st.subheader("🔍 Agent 2: Extraction")
            with st.expander("Extracted Fields", expanded=True):
                st.json(result["extracted_fields"])
        
        if result["validation_result"]:
            st.subheader("✅ Agent 3: Validation")
            validation = result["validation_result"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Valid", "Yes" if validation.get("is_valid") else "No")
            with col2:
                st.metric("Score", f"{validation.get('score', 0):.0%}")
            
            if validation.get("checks"):
                st.write("**Checks:**")
                for check in validation["checks"]:
                    icon = "✅" if check.get("status") == "pass" else "⚠️" if check.get("status") == "warning" else "❌"
                    st.write(f"  {icon} {check.get('rule', 'N/A')}: {check.get('details', '')}")
            
            if validation.get("issues"):
                st.error("**Issues:**")
                for issue in validation["issues"]:
                    st.write(f"  • {issue}")
            
            if validation.get("warnings"):
                st.warning("**Warnings:**")
                for warning in validation["warnings"]:
                    st.write(f"  • {warning}")
        
        if result["rules_applied"]:
            st.subheader("📚 RAG: Rules Applied")
            with st.expander(f"View {len(result['rules_applied'])} company rules"):
                for rule in result["rules_applied"]:
                    st.write(f"  • {rule}")
        
        if result["summary"] and result["summary"].get("title"):
            st.subheader("📊 Agent 4: Executive Summary")
            summary = result["summary"]
            st.write(f"**{summary['title']}**")
            if summary.get("one_liner"):
                st.info(summary["one_liner"])
            
            if summary.get("key_highlights"):
                st.write("**Key Highlights:**")
                for highlight in summary["key_highlights"]:
                    st.write(f"  • {highlight}")
            
            if summary.get("action_items"):
                st.write("**Action Items:**")
                for item in summary["action_items"]:
                    st.write(f"  • {item}")
            
            if summary.get("risks"):
                st.warning("**Risks:**")
                for risk in summary["risks"]:
                    st.write(f"  • {risk}")
        
        st.markdown("---")
        st.subheader("💾 Export Results")
        export_data = {
            "classification": {
                "type": result["doc_type"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"]
            },
            "extracted_fields": result["extracted_fields"],
            "validation": result["validation_result"],
            "summary": result["summary"],
            "rules_applied": result["rules_applied"],
            "pipeline_errors": result.get("pipeline_errors", [])
        }
        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(export_data, indent=2, default=str),
            file_name="document_processing_report.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
