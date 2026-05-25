# AgentEval-Framework

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
└──────┬──────────────────────────────────┬───────────┘
       │                                  │
┌──────▼──────┐                  ┌────────▼────────┐
│ spec_search │                  │ log_classifier  │
│   @tool     │                  │    @tool        │
│             │                  │                 │
│ Keyword RAG │                  │ Few-shot LLM    │
│ over NVMe,  │                  │ classifies logs │
│ PCIe, UFS   │                  │ INFO/WARN/CRIT  │
│ spec docs   │                  │                 │
└─────────────┘                  └─────────────────┘
```

**Three layers:**
- **Tool layer** — standalone Python functions decorated with `@tool`, each with a precise docstring the agent reads to decide when to invoke it
- **Orchestration layer** — LangGraph `StateGraph` with conditional routing between agent reasoning and tool execution
- **Integration layer** — FastAPI REST endpoint exposing the full pipeline via `POST /ask`

---

## Project Structure

```
AgentEval-Framework/
├── tools/
│   ├── spec_search.py        # RAG tool: searches SSD spec documents
│   ├── log_classifier.py     # Few-shot tool: classifies firmware logs
│   ├── code_explainer.py     # (planned) explains C/Verilog snippets
│   └── __init__.py
├── agent/
│   ├── graph.py              # LangGraph StateGraph definition
│   ├── state.py              # AgentState TypedDict
│   └── __init__.py
├── api/
│   ├── main.py               # FastAPI app with /ask endpoint
│   ├── schemas.py            # Pydantic request/response models
│   └── __init__.py
├── notebooks/
│   ├── 01_basic_llm_call.ipynb      # LLM API fundamentals
│   ├── 02_rag_pipeline.ipynb        # RAG pipeline walkthrough
│   └── 03_langgraph_agent.ipynb     # LangGraph agent demo
├── tests/
│   ├── test_spec_search.py          # Unit tests for spec search tool
│   ├── test_log_classifier.py       # Unit tests for log classifier
│   └── golden_qa.json               # Known Q&A pairs for eval
├── data/
│   └── pdfs/                        # Spec PDFs (not committed)
├── .env.example                     # Environment variable template
├── config.yaml                      # Model, retrieval, API config
├── requirements.txt                 # Pinned dependencies
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
python -m venv .venv311
# Windows:
.venv311\Scripts\activate
# Mac/Linux:
source .venv311/bin/activate

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
```

### 5. Run the full agent

```bash
python tools/run_agent.py
```


## Running Tests

```bash
pytest tests/ -v
```

---


## Tech Stack

| Component | Technology |
|---|---|
| LLM provider | Ollama (local) / Groq (cloud) |
| Agent orchestration | LangGraph |
| Tool framework | LangChain `@tool` |
| Vector search | FAISS (planned) |
| API layer | FastAPI + Pydantic |
| Embeddings | sentence-transformers (planned) |
| Testing | pytest |

---

## Background

This project was built during the **Micron International Academy** program, an academic-industrial collaboration between Politecnico di Milano and Micron Technology at the University of Naples Federico II. The goal was to explore how agentic AI can be integrated into real hardware engineering workflows — specifically for SSD ASIC architecture teams who spend significant time navigating technical specifications, reviewing firmware logs, and cross-referencing design documents.

---

## Author

**Subhadip Banerjee**  
MSc Computer Science and Engineering, Politecnico di Milano  
[LinkedIn](https://www.linkedin.com/in/subhadip-banerjee-355234183/) | [GitHub](https://github.com/Subhadip0904)
