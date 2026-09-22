# LLM Gateway & Multimodal Inference Platform

A production-oriented LLM Gateway & Multimodal Inference Platform built to demonstrate model serving, routing, multimodal inference, MCP interoperability, evaluation, observability, security, caching, and production deployment engineering.

## Target Architecture Overview

```text
                         USER / CLIENT
                              |
                              v
                    +---------------------+
                    |   FastAPI Gateway   |
                    +---------------------+
                              |
                              v
                    +---------------------+
                    | Request Validation  |
                    | & Security Layer    |
                    +---------------------+
                              |
                              v
                    +---------------------+
                    |   Model Router      |
                    | complexity/cost/etc. |
                    +---------------------+
                       /              \
                      /                \
                     v                  v
              +-----------+      +-------------+
              | Text LLM  |      | Vision LLM  |
              +-----------+      +-------------+
                     \                /
                      \              /
                       v            v
                    +---------------------+
                    |   vLLM Serving      |
                    +---------------------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
       Semantic Cache     MCP Client       Evaluation
             |                |             & Tracing
             |                |
             |        +-------+-------+
             |        |               |
             v        v               v
          pgvector  PostgreSQL     Documents
                     MCP Server     MCP Server
```

## Core Distinction

- **Multi-Tool AI Agent**: Focuses on agentic reasoning, planning, tool calling, and memory.
- **LLM Gateway Platform (This Repository)**: Focuses on LLM infrastructure, API gateway, model serving (vLLM), complexity/cost routing, multimodal input processing, Model Context Protocol (MCP) servers/client, evaluation (LLM-as-a-Judge), observability, security guardrails, semantic caching (pgvector), and production engineering.

## Request Lifecycle

How a single `POST /v1/chat/completions` call actually flows through the current implementation, file by file:

```text
CLIENT REQUEST
      |
      v
app/main.py                    assembles the FastAPI app, registers
                                middleware + exception handlers + routers
      |
      v
app/core/middleware.py         assigns/reads X-Request-ID, logs
                                incoming/completed request
      |
      v
app/api/v1/chat.py  <------->  app/schemas/chat.py
POST /v1/chat/completions      ChatCompletionRequest, ChatMessage,
(the HTTP endpoint)            ContentPart, ImageURL -- validates every
      |                        request BEFORE the function body runs
      |                              |
      |                              v (validation fails)
      |                        app/core/exceptions.py
      |                        formats the 422 error JSON
      v
app/cache/semantic_cache.py <-> app/cache/embeddings.py
lookup() before calling any     EmbeddingClient -> real OpenAI
model (HIT -> return early)     /embeddings call
      |
      v (MISS)
app/router/router.py           ModelRouter -- decides which
determine_route()               model, then orchestrates the call
execute_with_fallback()
      |------------------------------|
      v                              v
app/router/analyzer.py         app/router/schemas.py
analyze() -> complexity,        ComplexityLevel, RequestType,
request_type                    RoutingDecision
      |
      v
app/providers/registry.py      maps model name -> provider instance
get_provider_for_model()
      |----------------|----------------|
      v                v                v
providers/mock.py  openai_provider.py  vllm_provider.py
(fake response)    (real HTTP call     (real HTTP or
                     to OpenAI)         simulated)
      |----------------|----------------|
                        | .generate() -> real LLM response
                        | (raises providers/exceptions.py on failure,
                        |  caught by router.py for fallback retry)
                        v
app/router/router.py           calculate_cost() using real token
(back in execute_with_fallback) usage + model's configured rate
      |
      v
app/cache/semantic_cache.py    store() -- save this response for
                                future similar prompts
      |
      v
app/api/v1/chat.py             sets X-Cache-Hit, X-Response-Latency-Ms,
(back in the endpoint)         X-Estimated-Cost-USD headers, logs,
                                returns to client
      |
      v
CLIENT RESPONSE
```

Read by almost every file above: `app/config.py` (settings singleton loaded from `.env`) and `app/core/logging.py` (shared logger).

## Milestone Status

- [x] **Milestone 0 — Project Definition & Repository Setup**
- [x] **Milestone 1 — FastAPI LLM Gateway**
- [x] **Milestone 2 — Model Provider Abstraction**
- [x] **Milestone 3 — Model Router**
- [x] **Milestone 4 — vLLM Model Serving**
- [x] **Milestone 5 — Token, Latency & Cost Tracking**
- [x] **Milestone 6 — Semantic Cache**
- [x] **Milestone 7 — Multimodal Input Pipeline**
- [x] **Milestone 8 — RAG Pipeline**
- [x] **Milestone 9 — Advanced Retrieval**
- [ ] Milestone 10 — Custom PostgreSQL MCP Server
- [ ] Milestone 11 — Custom Document MCP Server
- [ ] Milestone 12 — MCP Client Integration
- [ ] Milestone 13 — Security & Guardrails
- [ ] Milestone 14 — Evaluation Framework
- [ ] Milestone 15 — Observability
- [ ] Milestone 16 — Cost & Latency Optimization
- [ ] Milestone 17 — Docker & Local Multi-Service Deployment
- [ ] Milestone 18 — Automated Testing & CI/CD
- [ ] Milestone 19 — GCP / Remote GPU Deployment
- [ ] Milestone 20 — Optional A2A Interoperability
- [ ] Milestone 21 — Final Documentation & Demo

## Getting Started

### Prerequisites

- Python 3.12+
- Virtual environment tool (`venv` or `virtualenv`)

### Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
uvicorn app.main:app --reload --port 8000
```

Access API docs at `http://localhost:8000/docs`, health check at `http://localhost:8000/health`, or send chat completions to `http://localhost:8000/v1/chat/completions`.

### Running Tests

```bash
pytest
```
