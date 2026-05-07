# 📄 PwC Agentic Document Processing

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Gemini-4285F4.svg)](https://cloud.google.com/vertex-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI-Powered Multimodal Document Intelligence — Classify, Extract, Validate, and Summarize documents using a 4-agent pipeline with Gemini 2.5 Flash, LangGraph orchestration, and RAG-augmented validation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                             │
│  (Paste Text · Upload PDF · Upload Image · Export JSON)     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Pipeline                         │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐  │
│  │ Agent 1  │──▶│ Agent 2  │──▶│ Agent 3  │──▶│Agent 4 │  │
│  │Classify  │   │Extract   │   │Validate  │   │Summarize│  │
│  └──────────┘   └────┬─────┘   └────┬─────┘   └────────┘  │
│                      │              │                       │
│                      ▼              ▼                       │
│               ┌─────────────────────────────┐               │
│               │   RAG Retriever (32 Rules)  │               │
│               │   Vector Search / Cosine    │               │
│               └─────────────────────────────┘               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Vertex AI · Gemini 2.5 Flash                    │
│     (Text Embeddings · Vector Search · Vision API)          │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Multi-Modal Input** — Paste text, upload PDFs, or upload images (OCR via Gemini Vision)
- **4-Agent Pipeline** — Classification → Extraction → Validation → Summary
- **RAG-Augmented** — 32 company rules across 4 document types with vector search
- **Robust JSON Parsing** — Handles malformed LLM outputs with multiple fallback strategies
- **Production Ready** — Multi-stage Docker build, health checks, non-root user
- **Extensible** — Abstract base agent class, pluggable RAG backends

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI API enabled

### 1. Clone & Setup

```bash
git clone <repository-url>
cd pwc-demo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Google Cloud settings
```

### 3. Run

```bash
streamlit run src/app.py
```

### 4. Docker

```bash
# Build
docker build --target production -t pwc-agentic-doc:latest .

# Run
docker run -p 8080:8080 --env-file .env pwc-agentic-doc:latest
```

## 📁 Project Structure

```
pwc-demo/
├── src/
│   ├── app.py                 # Thin entry point
│   ├── config.py              # Centralized settings
│   ├── models/state.py        # DocumentState TypedDict
│   ├── agents/                # 4 processing agents
│   │   ├── base.py            # BaseAgent (abstract)
│   │   ├── classifier.py      # Agent 1: Classification
│   │   ├── extractor.py       # Agent 2: Extraction + RAG
│   │   ├── validator.py       # Agent 3: Validation + RAG
│   │   └── summarizer.py      # Agent 4: Summary
│   ├── rag/                   # RAG components
│   │   ├── rules.py           # 32 company rules
│   │   ├── embeddings.py      # Embedding computation
│   │   └── retriever.py       # Vector search + cosine similarity
│   ├── pipeline/orchestrator.py  # LangGraph StateGraph
│   ├── utils/                 # Utilities
│   │   ├── json_parser.py     # Robust JSON parsing
│   │   └── file_handler.py    # PDF/image extraction
│   └── ui/                    # Streamlit UI
│       ├── components.py      # Reusable components
│       └── pages.py           # Main page layout
├── tests/                     # Test suite
├── docs/                      # Architecture & deployment docs
├── Dockerfile                 # Multi-stage production build
├── docker-compose.yml         # Local dev with hot-reload
└── pyproject.toml             # Modern Python packaging
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

## 📦 Supported Document Types

| Type | Rules | Fields Extracted |
|------|-------|-----------------|
| 📄 Invoice | 8 rules | Invoice #, date, vendor, amounts, tax, payment terms |
| 📋 Contract | 8 rules | Parties, dates, terms, compensation, clauses |
| 📊 Report | 8 rules | Title, period, metrics, trends, recommendations |
| ✉️ Email | 8 rules | Sender, recipient, subject, sentiment, urgency |

## 🔧 Configuration

All settings are centralized in `src/config.py` via a frozen dataclass:

| Setting | Default | Description |
|---------|---------|-------------|
| `project_id` | `pwc-agentic-demo` | Google Cloud project |
| `location` | `us-central1` | GCP region |
| `model_name` | `gemini-2.5-flash` | Gemini model |
| `embedding_model` | `gemini-embedding-001` | Embedding model |
| `max_retrieved_rules` | `8` | Top-K rules for RAG |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
