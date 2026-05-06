# Vertex AI Platform — Scalability & Model Comparison

> Last updated: May 2026 | Sources: Official GCP Documentation

---

## Table of Contents
1. [Vertex AI Rate Limits](#1-vertex-ai-rate-limits)
2. [Gemini Model Comparison](#2-gemini-model-comparison)
3. [All Models on Vertex AI](#3-all-models-on-vertex-ai)
4. [Pricing Comparison](#4-pricing-comparison)
5. [Capabilities Matrix](#5-capabilities-matrix)
6. [Auto-Scaling & Provisioned Throughput](#6-auto-scaling--provisioned-throughput)
7. [SLA & Uptime](#7-sla--uptime)
8. [Regional Availability](#8-regional-availability)
9. [Recommendations](#9-recommendations)

---

## 1. Vertex AI Rate Limits

### Usage Tiers (Gemini API)

| Tier | Qualification | Billing Cap |
|------|--------------|-------------|
| **Free** | Active project or free trial | N/A |
| **Tier 1** | Link active billing account | $250 |
| **Tier 2** | Paid $100 + 3 days | $2,000 |
| **Tier 3** | Paid $1,000 + 30 days | $20,000 - $100,000+ |

Rate limits measured across:
- **RPM** — Requests per Minute (varies by model/tier)
- **TPM** — Tokens per Minute
- **RPD** — Requests per Day (resets midnight PT)

### Batch API Limits

| Model | Max Enqueued Tokens | Concurrent Jobs |
|-------|-------------------|-----------------|
| Gemini 2.5 Pro | 5,000,000 | 100 (shared pool) |
| Gemini 2.5 Flash | 3,000,000 | 100 (shared pool) |
| Gemini 2.5 Flash-Lite | 10,000,000 | 100 (shared pool) |
| Gemini 2.0 Flash | 10,000,000 | 100 (shared pool) |

### Other Service Limits

| Service | Limit |
|---------|-------|
| Agent Engine queries/min | 90 |
| Concurrent BidiStream connections | 10 |
| RAG Engine data management | 60 RPM |
| Evaluation requests | 1,000 RPM |

---

## 2. Gemini Model Comparison

### Gemini 2.5 Family (Current Generation)

| Model | Context | Input $/1M | Output $/1M | Thinking | Multimodal | Best For |
|-------|---------|-----------|------------|----------|------------|----------|
| **Gemini 2.5 Pro** | 1M | $1.25 | $10.00 | Yes | Text/Image/Video/Audio | Complex reasoning, agentic |
| **Gemini 2.5 Flash** | 1M | $0.30 | $2.50 | Yes | Text/Image/Video/Audio | Best balance (our model) |
| **Gemini 2.5 Flash-Lite** | 1M | $0.10 | $0.40 | No | Text/Image/Video | Cost-optimized high volume |
| **Gemini 2.5 Flash Image** | 1M | $0.30 | $2.50 | Yes | Text/Image | Image generation |

### Gemini 2.0 Family (Previous Generation)

| Model | Context | Input $/1M | Output $/1M | Multimodal | Best For |
|-------|---------|-----------|------------|------------|----------|
| **Gemini 2.0 Flash** | 1M | $0.15 | $0.60 | Text/Image/Video/Audio | General purpose |
| **Gemini 2.0 Flash-Lite** | 1M | $0.075 | $0.30 | Text/Image | Ultra-cheap |

### Gemini 3.x (Preview — Next Generation)

| Model | Context | Input $/1M | Output $/1M | Key Improvement |
|-------|---------|-----------|------------|-----------------|
| **Gemini 3.1 Pro Preview** | 1M | $2.00 | $12.00 | Advanced reasoning |
| **Gemini 3 Flash Preview** | 1M | $0.50 | $3.00 | Best multimodal, near-zero thinking |
| **Gemini 3.1 Flash-Lite Preview** | 1M | $0.25 | $1.50 | Most cost-efficient |

---

## 3. All Models on Vertex AI

### Anthropic Claude

| Model | Input $/1M | Output $/1M | Batch Input | Batch Output | Key Features |
|-------|-----------|------------|-------------|--------------|--------------|
| **Claude Opus 4.7** | $5.00 | $25.00 | $2.50 | $12.50 | Top-tier coding, agents |
| **Claude Sonnet 4.6** | $3.00 | $15.00 | $1.50 | $7.50 | Balanced coding/agents |
| **Claude Opus 4.5** | $5.00 | $25.00 | $2.50 | $12.50 | Coding, computer use |
| **Claude Sonnet 4.5** | $3.00 | $15.00 | $1.50 | $7.50 | Real-world agents |
| **Claude Haiku 4.5** | $1.00 | $5.00 | $0.50 | $2.50 | Fast, cost-effective |

### Meta Llama

| Model | Input $/1M | Output $/1M | Batch Input | Batch Output | Key Features |
|-------|-----------|------------|-------------|--------------|--------------|
| **Llama 4 Maverick** | $0.35 | $1.15 | $0.175 | $0.575 | 400B MoE, multimodal |
| **Llama 4 Scout** | $0.25 | $0.70 | $0.125 | $0.35 | 109B MoE, long context |
| **Llama 3.3 70B** | $0.72 | $0.72 | $0.36 | $0.36 | Text-only, instruction-tuned |

### Mistral

| Model | Input $/1M | Output $/1M | Key Features |
|-------|-----------|------------|--------------|
| **Mistral Medium 3** | $0.40 | $2.00 | Programming, math, 80+ languages |
| **Mistral Small 3.1** | $0.10 | $0.30 | Low-latency, 128K context |
| **Mistral OCR** | $0.0005/page | $0.0005/page | Document OCR, tables, equations |
| **Codestral 2** | $0.30 | $0.90 | Code generation, FIM completion |

### xAI Grok

| Model | Input $/1M | Output $/1M | Cache Hit |
|-------|-----------|------------|-----------|
| **Grok 4.20 Reasoning** | $2.00 | $6.00 | $0.20 |
| **Grok 4.1 Fast** | $0.20 | $0.50 | $0.05 |

### DeepSeek

| Model | Input $/1M | Output $/1M | Batch Input | Batch Output |
|-------|-----------|------------|-------------|--------------|
| **DeepSeek-V3.2** | $0.56 | $1.68 | $0.28 | $0.84 |
| **DeepSeek-V3.1** | $0.60 | $1.70 | $0.30 | $0.85 |
| **DeepSeek-R1** | $1.35 | $5.40 | $0.675 | $2.70 |
| **DeepSeek-OCR** | $0.30/page | $1.20/page | - | - |

### Qwen

| Model | Input $/1M | Output $/1M |
|-------|-----------|------------|
| **Qwen3-Next-80B** | $0.15 | $1.20 |
| **Qwen3-Coder-480B** | $0.22 | $1.80 |
| **Qwen3-235B** | $0.22 | $0.88 |

### OpenAI (Open Models)

| Model | Input $/1M | Output $/1M |
|-------|-----------|------------|
| **gpt-oss-120b** | $0.09 | $0.36 |
| **gpt-oss-20b** | $0.07 | $0.25 |

### Others

| Provider | Model | Input $/1M | Output $/1M |
|----------|-------|-----------|------------|
| MiniMax | M2 | $0.30 | $1.20 |
| Moonshot | Kimi-K2-Thinking | $0.60 | $2.50 |
| GLM | GLM-5 | $1.00 | $3.20 |
| Google Gemma | Gemma 4 26B | $0.15 | $0.60 |

---

## 4. Pricing Comparison (Ranked by Cost)

### Cheapest Input Tokens

| Rank | Model | Input $/1M | Output $/1M | Provider |
|------|-------|-----------|------------|----------|
| 1 | gpt-oss-20b | $0.07 | $0.25 | OpenAI |
| 2 | Gemini 2.0 Flash-Lite | $0.075 | $0.30 | Google |
| 3 | gpt-oss-120b | $0.09 | $0.36 | OpenAI |
| 4 | Gemini 2.5 Flash-Lite | $0.10 | $0.40 | Google |
| 5 | Mistral Small 3.1 | $0.10 | $0.30 | Mistral |
| 6 | Qwen3-Next-80B | $0.15 | $1.20 | Qwen |
| 7 | Gemini 2.0 Flash | $0.15 | $0.60 | Google |
| 8 | Gemma 4 26B | $0.15 | $0.60 | Google |
| 9 | Gemini 2.5 Flash | $0.30 | $2.50 | Google |
| 10 | Llama 4 Scout | $0.25 | $0.70 | Meta |

### Cheapest Output Tokens

| Rank | Model | Input $/1M | Output $/1M | Provider |
|------|-------|-----------|------------|----------|
| 1 | gpt-oss-20b | $0.07 | $0.25 | OpenAI |
| 2 | Gemini 2.0 Flash-Lite | $0.075 | $0.30 | Google |
| 3 | Mistral Small 3.1 | $0.10 | $0.30 | Mistral |
| 4 | gpt-oss-120b | $0.09 | $0.36 | OpenAI |
| 5 | Gemini 2.5 Flash-Lite | $0.10 | $0.40 | Google |

### Batch Discount Pricing (~50% off)

| Model | Batch Input | Batch Output |
|-------|------------|-------------|
| Gemini 2.5 Flash | $0.15 | $1.25 |
| Gemini 2.5 Pro | $0.625 | $5.00 |
| Claude Sonnet 4.5 | $1.50 | $7.50 |
| Llama 4 Maverick | $0.175 | $0.575 |

---

## 5. Capabilities Matrix

### Gemini Models

| Capability | 2.5 Pro | 2.5 Flash | 2.5 Flash-Lite | 2.0 Flash | 2.0 Flash-Lite |
|------------|---------|-----------|----------------|-----------|----------------|
| Multimodal Input | Text/Image/Video/Audio | Text/Image/Video/Audio | Text/Image/Video | Text/Image/Video/Audio | Text/Image |
| Structured Output (JSON) | Yes | Yes | Yes | Yes | Yes |
| Function Calling | Yes | Yes | Yes | Yes | Yes |
| Grounding (Google Search) | Yes | Yes | Yes | Yes | Yes |
| Code Execution | Yes | Yes | Yes | Yes | Yes |
| System Instructions | Yes | Yes | Yes | Yes | Yes |
| Thinking/Reasoning | Yes | Yes | No | No | No |
| Context Caching | Yes | Yes | Yes | Yes | Yes |
| Batch Prediction | Yes | Yes | Yes | Yes | Yes |
| Live API (Streaming) | No | Yes | No | Yes | No |
| Tuning Cost/1M tokens | $25 | $5 | $1.50 | $3 | $1 |

### Partner Models

| Capability | Claude Opus 4.7 | Claude Sonnet 4.5 | Llama 4 Maverick | Mistral Medium 3 | DeepSeek-V3.2 |
|------------|----------------|-------------------|------------------|------------------|---------------|
| Multimodal | Text + Vision | Text + Vision | Text + Image | Text + Image | Text + Vision |
| Structured Output | Yes | Yes | Yes | Yes | Yes |
| Function Calling | Yes | Yes | Yes | Yes | Yes |
| Computer Use | Yes | Yes | No | No | No |
| Web Search | Yes | Yes | No | No | No |
| Batch | Yes | Yes | No | No | Yes |
| Context Window | 200K | 200K | Long | 128K | 128K |

---

## 6. Auto-Scaling & Provisioned Throughput

### Pay-As-You-Go (PayGo)
- No upfront commitment
- Auto-scales based on demand (subject to quotas)
- Charged in 30-second increments
- Best for: Development, variable workloads

### Batch Prediction
- **Gemini models:** Fully auto-scaled via shared pool
- 100 concurrent batch requests
- 50% discount on all token prices
- Best for: Non-urgent, high-volume processing

### Provisioned Throughput (Guaranteed Capacity)

| Commitment | Price per GSU/month |
|------------|---------------------|
| 1 week | $1,200 |
| 1 month | $2,700 |
| 3 months | $2,400 |
| 1 year | $2,000 |

Best for: Production workloads requiring guaranteed latency

### Model Optimizer (Experimental)
- Single meta-endpoint routes to best-fit model
- Three modes: **Cost**, **Balanced**, **Quality**
- Dynamic pricing based on model selected

---

## 7. SLA & Uptime

| Service | Monthly Uptime |
|---------|----------------|
| Training/Deployment/Batch | 99.9% |
| Custom Model Online (2+ nodes) | 99.5% |
| Vertex Pipelines | 99.5% |

### Financial Credits (if SLO not met)

| Uptime | Credit |
|--------|--------|
| 99% to < 99.9% | 10% of monthly bill |
| 95% to < 99% | 25% of monthly bill |
| < 95% | 50% of monthly bill |

---

## 8. Regional Availability

### Key Regions

| Region | ID | Best For |
|--------|-----|----------|
| Iowa (US) | us-central1 | **Primary — most models available** |
| South Carolina | us-east1 | US East |
| Oregon | us-west1 | US West |
| Mumbai | asia-south1 | India |
| Singapore | asia-southeast1 | APAC |
| London | europe-west2 | UK/EU |
| Frankfurt | europe-west3 | EU |
| Tokyo | asia-northeast1 | Japan |

### Multi-Region Options
- **Global** — routes to nearest region (recommended for PayGo)
- **US Multi-Region** — US-only data residency
- **EU Multi-Region** — EU-only data residency

---

## 9. Recommendations

### For PwC Demo (Current Setup)
- **Model:** Gemini 2.5 Flash — excellent balance of cost ($0.30/$2.50) and capability
- **Region:** us-central1 — maximum feature availability
- **Pricing:** PayGo — no commitment needed for demo

### For Enterprise Scale (10K+ docs/day)
- **Model:** Gemini 2.5 Flash for classification/extraction, 2.5 Pro for complex reasoning
- **Batch:** Use Batch API for non-urgent workloads (50% discount)
- **Caching:** Context caching for repeated prompts (90% cost reduction)
- **Provisioned:** Purchase GSUs for guaranteed throughput

### Cost Optimization Strategies
1. **Context caching** — 90% off for repeated system prompts ($0.03/M vs $0.30/M)
2. **Batch processing** — 50% off all token prices
3. **Flash-Lite for simple tasks** — $0.10/$0.40 vs $0.30/$2.50
4. **Model Optimizer** — auto-routes to cheapest suitable model

### Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| Document classification | Gemini 2.5 Flash | Fast, cheap, accurate |
| Data extraction | Gemini 2.5 Flash | Structured output, good accuracy |
| Complex validation | Gemini 2.5 Pro | Better reasoning |
| High-volume simple tasks | Gemini 2.5 Flash-Lite | 3x cheaper |
| OCR/Document understanding | Mistral OCR | Purpose-built, $0.0005/page |
| Code-heavy analysis | Claude Sonnet 4.5 | Best coding capability |
| Budget-constrained | Gemini 2.0 Flash-Lite | $0.075/$0.30 |
