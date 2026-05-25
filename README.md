# AgentEval-Framework

[![Tests](https://github.com/Subhadip0904/AgentEval-Framework/actions/workflows/test.yml/badge.svg)](https://github.com/Subhadip0904/AgentEval-Framework/actions/workflows/test.yml)
[![Lint](https://github.com/Subhadip0904/AgentEval-Framework/actions/workflows/lint.yml/badge.svg)](https://github.com/Subhadip0904/AgentEval-Framework/actions/workflows/lint.yml)

> A composable, modular AI engineering assistant built with LangChain, LangGraph, and FastAPI — designed to help SSD ASIC architecture teams query technical specifications, classify firmware logs, and automate repetitive engineering workflows through natural language.

---

## What This Is

AgentEval-Framework is a prototype agentic AI assistant that demonstrates the **tool + skill** architecture pattern for engineering environments. Instead of a single monolithic chatbot, it is built as a set of independent, composable AI modules — each doing one focused task — orchestrated by a LangGraph agent that decides which tool to call based on the engineer's natural language query.

Built as part of the **Micron International Academy** program at the University of Naples Federico II, this project applies AI to real SSD architecture engineering workflows.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI REST Layer                  │
│              POST /ask  →  GET /health               │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│              LangGraph Orchestration Agent           │
│   StateGraph → agent_node → tools_condition → tools  │
└────┬─────────────────────────────────┬─────────────┬┘
     │                                 │             │
┌────▼──────┐          ┌──────────┐    │    ┌────────▼────────┐
│spec_search│          │FAISS Idx │    │    │code_explainer   │
│   @tool   │─────────→│ Semantic │    │    │    @tool        │
│           │          │  Search  │    │    │                 │
│Vector RAG │          │          │    │    │Firmware code    │
│over Specs │          └──────────┘    │    │explanation      │
└───────────┘                          │    └─────────────────┘
                          ┌────────────▼────────┐
                          │log_classifier       │
                          │    @tool            │
                          │                     │
                          │ Few-shot LLM        │
                          │ classifies logs     │
                          │ INFO/WARN/CRIT      │
                          └─────────────────────┘
```

**Three layers:**
- **Tool layer** — three independent tools: `spec_search` (semantic RAG), `log_classifier` (few-shot), `code_explainer` (context-aware)
- **Orchestration layer** — LangGraph `StateGraph` with conditional routing and tool execution
- **Integration layer** — FastAPI REST endpoint exposing the full pipeline via `POST /ask`

---

## Project Structure

```
AgentEval-Framework/
├── tools/
│   ├── spec_search.py        # RAG tool: semantic search over spec PDFs
│   ├── log_classifier.py     # Few-shot tool: classifies firmware logs (INFO/WARN/CRITICAL)
│   ├── code_explainer.py     # Explains C/Verilog code in hardware context
│   ├── pdf_loader.py         # Loads and chunks PDF specifications
│   ├── retriever.py          # FAISS vector search over documents
│   └── __init__.py
├── agent/
│   ├── graph.py              # LangGraph StateGraph orchestration
│   ├── state.py              # AgentState TypedDict definition
│   └── __init__.py
├── api/
│   ├── main.py               # FastAPI app with /ask endpoint
│   ├── schemas.py            # Pydantic request/response models
│   └── __init__.py
├── evals/
│   ├── evaluator.py          # Framework for evaluating agent accuracy
│   └── __init__.py
├── tests/
│   ├── test_spec_search.py          # Unit tests for spec search
│   ├── test_log_classifier.py       # Unit tests for log classifier
│   ├── test_api.py                  # Integration tests for API
│   └── golden_qa.json               # Known Q&A pairs for evaluation
├── .github/
│   └── workflows/
│       ├── test.yml          # GitHub Actions: run pytest
│       └── lint.yml          # GitHub Actions: black, isort, flake8
├── data/
│   ├── pdfs/                        # Spec PDFs (add your own)
│   ├── golden_qa.json               # Golden Q&A pairs
│   └── faiss_index                  # Built FAISS index
├── notebooks/
│   ├── 01_basic_llm_call.ipynb
│   ├── 02_rag_pipeline.ipynb
│   └── 03_langgraph_agent.ipynb
├── .env.example                     # Environment template
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11
- [Ollama](https://ollama.com) installed locally (free, no API key needed)

### 1. Clone and set up environment

```bash
git clone https://github.com/Subhadip0904/AgentEval-Framework.git
cd AgentEval-Framework
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Start Ollama (local LLM — no API key required)

```bash
ollama pull llama3.2
ollama serve
```

Leave this terminal open. Ollama listens on `http://localhost:11434`.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if using a cloud provider instead of Ollama
```

### 4. Run the tools directly

```bash
# Test spec search
python tools/spec_search.py

# Test log classifier
python tools/log_classifier.py

# Test code explainer
python tools/code_explainer.py
```

### 5. Run the API server

```bash
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs for interactive API docs
```

### 6. Test the API

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How does NVMe queue depth work?"}'
```

### 7. Run tests and evaluation

```bash
# Run unit tests
pytest tests/ -v

# Run evaluation on golden Q&A pairs
python -m evals.evaluator
```

---


## Tech Stack

| Component | Technology |
|---|---|
| LLM provider | Ollama (local) or Groq/OpenAI (cloud) |
| Agent orchestration | LangGraph |
| Tool framework | LangChain `@tool` decorator |
| Vector search | FAISS (semantic similarity) |
| API layer | FastAPI + Pydantic |
| Embeddings | sentence-transformers |
| PDF processing | pdfplumber + LangChain splitter |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## Background

This project was built during the **Micron International Academy** program, an academic-industrial collaboration between Politecnico di Milano and Micron Technology at the University of Naples Federico II. The goal was to explore how agentic AI can be integrated into real hardware engineering workflows — specifically for SSD ASIC architecture teams who spend significant time navigating technical specifications, reviewing firmware logs, and cross-referencing design documents.

---

## Author

**Subhadip Banerjee**  
MSc Computer Science and Engineering, Politecnico di Milano  
[LinkedIn](https://www.linkedin.com/in/subhadip-banerjee-355234183/) | [GitHub](https://github.com/Subhadip0904)
