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
app/core/rate_limit.py         enforce_rate_limit -- in-memory sliding
enforce_rate_limit()           window per client IP, runs first so even
      |                        invalid-key attempts get throttled
      v (limit ok)
app/core/security.py           verify_api_key -- no-op if GATEWAY_API_KEY
verify_api_key()                is unset, else requires a matching Bearer
      |                        token
      v (auth ok)
app/api/v1/chat.py  <------->  app/schemas/chat.py
POST /v1/chat/completions      ChatCompletionRequest, ChatMessage,
(the HTTP endpoint)            ContentPart, ImageURL -- validates every
      |                        request (incl. size/message-count caps)
      |                        BEFORE the function body runs
      |                              |
      |                              v (validation fails)
      |                        app/core/exceptions.py
      |                        formats the 422 error JSON,
      |                        records it in app/core/metrics.py
      v
app/cache/semantic_cache.py <-> app/cache/embeddings.py
lookup() before calling any     EmbeddingClient -> real OpenAI
model (HIT -> return early,     /embeddings call
skipped entirely if use_rag)
      |
      v (MISS)
app/rag/augment.py             only if payload.use_rag is true --
augment_with_context()          retrieves top-k chunks from
      |                        app/rag/document_store.py, prepends
      |                        them as a system message
      v
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
                                records the request in
                                app/core/metrics.py, returns to client
      |
      v
CLIENT RESPONSE
```

Read by almost every file above: `app/config.py` (settings singleton loaded from `.env`) and `app/core/logging.py` (shared logger).

`POST /v1/documents` follows the same rate-limit/auth/validation gate, then goes straight to `app/rag/document_store.py`'s `ingest_document()` (no router/provider involved — it's chunk, embed, store).

Set `"stream": true` on a chat request to get Server-Sent Events instead of a single JSON response — each provider yields incremental `ChatCompletionChunk`s (real SSE passthrough for OpenAI/vLLM, simulated word-by-word for mock/GPU-less local dev) via `stream_generate()`. Streaming bypasses the semantic cache entirely (an accumulate-then-cache path added complexity for uncertain benefit) and, since HTTP headers are sent before the body starts, `X-Response-Latency-Ms`/`X-Estimated-Cost-USD` aren't available as headers for a streamed response — that data is logged and recorded in `/metrics` once the stream completes instead. Fallback on a streaming failure only happens *before* the first chunk is sent; once real data has reached the client it can't be un-sent, so a mid-stream failure is not recovered.

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
- [x] **Milestone 10 — Custom PostgreSQL MCP Server**
- [x] **Milestone 11 — Custom Document MCP Server**
- [x] **Milestone 12 — MCP Client Integration**
- [x] **Milestone 13 — Security & Guardrails**
- [x] **Milestone 14 — Evaluation Framework**
- [x] **Milestone 15 — Observability**
- [x] **Milestone 16 — Cost & Latency Optimization**
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

Access API docs at `http://localhost:8000/docs`, health check at `http://localhost:8000/health`, send chat completions to `http://localhost:8000/v1/chat/completions`, or ingest a document for RAG retrieval via `POST http://localhost:8000/v1/documents` (`{"text": "..."}`).

By default there's no authentication — set `GATEWAY_API_KEY` in `.env` to require `Authorization: Bearer <key>` on every `/v1/*` request (a startup warning is logged if it's left unset). Requests are also capped at `RATE_LIMIT_REQUESTS` per `RATE_LIMIT_WINDOW_SECONDS` per client, and payload size limits (`MAX_MESSAGES_PER_REQUEST`, `MAX_TOTAL_CONTENT_CHARS`, `MAX_DOCUMENT_CHARS`) reject pathologically large requests before they reach a model.

`POST /v1/evaluate` scores a `{prompt, response}` pair via LLM-as-a-Judge (reference-free — relevance/coherence/helpfulness by default, or pass a custom `criteria` list), calling `EVAL_JUDGE_MODEL` through the same router/provider infrastructure as chat completions. It's a separate, opt-in call — never run automatically on a chat completion, since judging is itself a real LLM call.

`GET /metrics` exposes request counts, cost, latency, cache hit rate, and errors by type in Prometheus text format (unauthenticated, like `/health`, for scrapers). Logs are structured JSON lines rather than plain text, for log-aggregation tooling.

### Running the PostgreSQL MCP Server

A separate process from the gateway — exposes read-only database access (`query_database`, `list_schema`) as MCP tools over stdio, backed by SQLite for now (`MCP_DB_PATH` in `.env`):

```bash
python -m app.mcp.postgres_server
```

### Running the Document MCP Server

Also a separate process — exposes `search_documents` and `ingest_document` as MCP tools over the same in-memory `DocumentStore` the chat endpoint's `use_rag` flag reads from:

```bash
python -m app.mcp.document_server
```

### Calling an MCP Server's Tools

`app/mcp/client.py` provides a reusable `MCPClient` that spawns a server module as a subprocess and speaks the real MCP protocol over stdio — useful for scripting or verifying a server end-to-end. It is intentionally standalone: it is not wired into `/v1/chat/completions`, since giving the model a live tool-calling loop mid-request would make this gateway agentic, which is out of scope (see Core Distinction above).

```python
from app.mcp.client import MCPClient

client = MCPClient("app.mcp.postgres_server")
tools = await client.list_tools()                                  # ["query_database", "list_schema"]
rows = await client.call_tool("query_database", {"sql": "SELECT 1"})
```

### Running Tests

```bash
pytest
```
