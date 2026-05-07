# Architecture — PwC Agentic Document Processing

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ Paste    │ │ PDF      │ │ Image    │ │ Export       │   │
│  │ Text     │ │ Upload   │ │ Upload   │ │ JSON         │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────────────┘   │
│       └─────────────┼───────────┘                            │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DocumentState (TypedDict)                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │
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
│               ┌──────────┐  ┌──────────┐                   │
│               │   RAG    │  │   RAG    │                   │
│               │Retriever │  │Retriever │                   │
│               └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Vertex AI / Gemini 2.5 Flash                  │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐    │
│  │ Text Embedding│  │ Vector Search  │  │ Vision API   │    │
│  │ (gemini-      │  │ (Matching      │  │ (Multimodal) │    │
│  │  embedding)   │  │  Engine)       │  │              │    │
│  └──────────────┘  └────────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### Stage 1: Classification
- **Agent:** `ClassificationAgent`
- **Input:** Raw document text
- **Output:** Document type (Invoice/Contract/Report/Email/Unknown), confidence, reasoning
- **Temperature:** 0.1 (deterministic)

### Stage 2: Extraction
- **Agent:** `ExtractionAgent`
- **Input:** Document text + classified type
- **Process:** RAG retrieval of relevant rules → LLM extraction with rules context
- **Output:** Key-value field extraction
- **Temperature:** 0.1

### Stage 3: Validation
- **Agent:** `ValidationAgent`
- **Input:** Extracted fields + document type + applied rules
- **Output:** is_valid, score, checks, issues, warnings
- **Temperature:** 0.1

### Stage 4: Summarization
- **Agent:** `SummarizationAgent`
- **Input:** Extracted fields + validation result
- **Output:** Title, one-liner, highlights, action items, risks, overall status
- **Temperature:** 0.3 (slightly more creative)

## RAG Architecture

```
Query Text ──▶ Embedding ──▶ Vector Search ──▶ Top-K Rules
    │                            │
    │                    Vertex AI Matching Engine
    │                    (or in-memory cosine similarity)
    │
    ▼
Retrieved Rules ──▶ LLM Prompt Context
```

## Module Structure

```
src/
├── config.py          # Settings (frozen dataclass)
├── app.py             # Thin entry point
├── models/state.py    # DocumentState TypedDict
├── agents/            # 4 processing agents + base class
├── rag/               # Rules, embeddings, retriever
├── pipeline/          # LangGraph orchestrator
├── utils/             # JSON parser, file handlers
└── ui/                # Streamlit components + pages
```
