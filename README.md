# Code Generator — LangGraph Agentic Pipeline
<img width="1890" height="835" alt="code-generator-langgraph" src="https://github.com/user-attachments/assets/8d6aea35-bda5-45e0-897d-62b20a7c6854" />

An agentic, self-correcting code generation system built with **LangGraph**, combining query understanding, retrieval-augmented generation (RAG), code validation, and multi-provider LLM fallback into a single orchestrated workflow — with a Streamlit chat interface and full LangSmith tracing.

🔗 **Live App:** [code-generator-langgraph.onrender.com](https://code-generator-langgraph.onrender.com/)
📦 **Repo:** [github.com/anujr-shri/code-generator-langgraph](https://github.com/anujr-shri/code-generator-langgraph)

---

## Overview

This project implements a multi-step agentic pipeline that takes a natural-language coding query, classifies and rewrites it, retrieves relevant context from a vector store, generates code via an LLM, validates the output (AST/compilation checks), and — if validation fails — feeds the error back into the LLM for a self-correcting retry loop.

The system is designed to keep working even if a primary LLM provider goes down, by falling back through a chain of providers automatically.

---

## Key Features

- **Agentic Workflow (LangGraph)** — a stateful graph orchestrates query classification → query rewriting → retrieval → code generation → validation → self-correction.
- **RAG-based Generation** — relevant code/context is retrieved from a **ChromaDB** vector store using `sentence-transformers` embeddings before generation.
- **Self-Correcting Loop** — generated code is validated (AST parsing / compilation checks); on failure, the execution error is fed back to the LLM for an automatic retry.
- **Multi-Model Fallback Chain** — if the primary provider fails or rate-limits, the pipeline automatically falls back: **Groq → Gemini → HuggingFace Inference**.
- **LangSmith Tracing** — every node, LLM call, and retry is traced for observability and debugging.
- **Streamlit Frontend** — multi-chat interface with session-based (UUID) chat threads, thread naming/renaming, and proper code-block rendering.
- **Deployed on Render** — live, publicly accessible deployment.

---

## Architecture

```
User Query
   │
   ▼
Query Classification Node ──► determines query type (code-gen, explanation, etc.)
   │
   ▼
Query Rewriting Node ──► reformulates query for better retrieval
   │
   ▼
Retrieval Node (ChromaDB + sentence-transformers) ──► fetches relevant context
   │
   ▼
Code Generation Node ──► LLM (Groq → Gemini → HuggingFace fallback)
   │
   ▼
Validation Node ──► AST / compilation check
   │
   ├── Pass ──► Return result to user
   │
   └── Fail ──► feed error back into Code Generation Node (self-correction loop)
```

All steps are traced end-to-end via **LangSmith**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLM Providers | Groq, Google Gemini, HuggingFace Inference (fallback chain) |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers |
| Validation | Python `ast` / compilation checks |
| Observability | LangSmith |
| Frontend | Streamlit |
| Deployment | Render |

---

## Project Structure

```
code-generator-langgraph/
├── chroma_code_genrator_db/   # Persisted ChromaDB vector store
├── frontend/                  # Streamlit chat application
├── tools/                     # LangGraph tools / node-level helper functions
├── utils/                     # Shared utility functions (state, helpers)
├── __init__.py
├── code_explanation_prompt.txt   # Prompt template for code explanation
├── llm_query.txt                 # Prompt template for LLM code-gen queries
├── query_type.txt                # Prompt template for query classification
├── retriver_prompt.txt           # Prompt template for retrieval step
├── rewrite_query.txt             # Prompt template for query rewriting
├── test_call_prompt.txt          # Prompt template for test/validation calls
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- API keys for at least one of: Groq, Google Gemini, HuggingFace

### Installation

```bash
git clone https://github.com/anujr-shri/code-generator-langgraph.git
cd code-generator-langgraph
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root with the following keys:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-generator-langgraph
```

### Running Locally

```bash
streamlit run frontend/frontend_multichat.py
```

The app will be available at `http://localhost:8501`.

---

## How It Works

1. **Query Classification** — The incoming user message is classified to determine the intent (e.g., generate new code, explain existing code, fix/debug code).
2. **Query Rewriting** — The query is rewritten/expanded to improve retrieval quality from the vector store.
3. **Retrieval (RAG)** — ChromaDB is queried using sentence-transformer embeddings to fetch relevant code snippets or documentation context.
4. **Code Generation** — The LLM generates code using the retrieved context. If the primary provider (Groq) is unavailable or fails, the request automatically falls back to Gemini, then HuggingFace.
5. **Validation** — The generated code is parsed/compiled to check for syntax errors.
6. **Self-Correction Loop** — If validation fails, the error message is appended to the context and sent back to the LLM for a corrected generation, up to a bounded number of retries.
7. **Response** — The final, validated code (or explanation) is returned to the user via the Streamlit chat interface.

---

## Observability

All LangGraph node executions, LLM calls, retries, and fallback transitions are traced via **LangSmith**, enabling step-by-step debugging of the agent's reasoning and execution path.

---

## Deployment

The app is deployed on **Render** with the Streamlit frontend served directly. Key deployment considerations addressed:

- Reduced deployment image size by managing indirect heavy dependencies (e.g., `torch`/`torchvision` pulled in via `sentence-transformers`).
- Proper Python package structure (`__init__.py`) for module resolution in production.
- HuggingFace Inference provider compatibility checks for model selection at runtime.

---

## Roadmap / Possible Improvements

- [ ] Add support for additional languages beyond Python in validation
- [ ] Expand fallback chain with additional providers
- [ ] Add unit tests for each LangGraph node
- [ ] Add user authentication for persistent chat history across sessions
- [ ] Caching layer for repeated retrieval queries

---

## Author

**Anuj Shrivastava (anujr-shri)**
CSE Student, IIIT Bhopal — ML Engineering enthusiast (LLM pipelines, RAG, agentic workflows)

---

