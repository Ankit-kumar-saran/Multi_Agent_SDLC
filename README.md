# Agentic RE-SDLC — Lightweight Financial Requirements Engineering & SDLC Advisor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-brightgreen.svg)]()
[![LLM: Google Gemini API](https://img.shields.io/badge/LLM-Google%20Gemini%20API-blue.svg)]()
[![Vector DB: Local Qdrant](https://img.shields.io/badge/Vector%20DB-Local%20Qdrant-red.svg)]()

A lightweight multi-agent platform for **financial software requirements engineering and SDLC model selection**, powered by the **Google Gemini API**, **Zero-Trust security & PII-masking guardrails**, **local Qdrant RAG**, and **human-in-the-loop approval gates**.

---

## Architecture Overview

| Component | File | Responsibility |
|---|---|---|
| **LLM Gateway** | [`agents/llm_client.py`](agents/llm_client.py) | Calls Google Gemini REST API (`GEMINI_API_KEY`, configurable primary + fallback models). |
| **Guardrails & Privacy** | [`guardrails/input_filter.py`](guardrails/input_filter.py), [`guardrails/output_filter.py`](guardrails/output_filter.py) | Blocks prompt injection/unsafe commands, masks PII/secrets before external API calls, and gates financial domain scope. |
| **Local RAG Pipeline** | [`rag/ingest.py`](rag/ingest.py), [`rag/retriever.py`](rag/retriever.py) | Sentence/paragraph-aware chunking + local CPU embeddings (`BAAI/bge-small-en-v1.5`) stored in local Qdrant (`qdrant_storage/` or server `6333`). |
| **Extraction & Clarification** | [`agents/extraction_agent.py`](agents/extraction_agent.py) | Dynamically extracts Functional & Non-Functional requirements, risk scores, controls, regulations, and clarification questions (no hardcoded templates). |
| **SDLC Selection Agent** | [`agents/sdlc_agent.py`](agents/sdlc_agent.py) | Evaluates project decision factors, ranks SDLC methodologies, produces a tailored SDLC workflow, and critiques human overrides. |
| **Orchestrator & PDF** | [`requirements_analysis.py`](requirements_analysis.py), [`pdf_generator.py`](pdf_generator.py) | Coordinates guardrails, RAG, LLM agents, traceability matrix, and SRS PDF generation. |
| **Web UI & API** | [`ui/app.py`](ui/app.py), [`backend/main.py`](backend/main.py), [`db/database.py`](db/database.py) | Streamlit chat interface, FastAPI endpoints, and PostgreSQL (with automatic SQLite fallback) persistence. |

---

## Quickstart

### 1. Environment Setup

```bash
cd Agentic-RE-SDLC
cp .env.example .env
# Set GEMINI_API_KEY in .env
```

### 2. Index Local RAG Knowledge Base

```bash
./venv/bin/python -m rag.ingest
```

### 3. Run Streamlit UI or FastAPI Backend

```bash
# Streamlit Web UI (Port 8501)
env PYTHONPATH=. ./venv/bin/streamlit run ui/app.py --server.port 8501

# Optional FastAPI Backend (Port 8000)
env PYTHONPATH=. ./venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 4. Run End-to-End Evaluation

```bash
./venv/bin/python scripts/evaluate_system.py
```

---

