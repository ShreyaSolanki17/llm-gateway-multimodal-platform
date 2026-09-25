# LLM Gateway & Multimodal Inference Platform

[![CI](https://github.com/ShreyaSolanki17/llm-gateway-multimodal-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/ShreyaSolanki17/llm-gateway-multimodal-platform/actions/workflows/ci.yml)

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

## Demo

Real captured output from a local run (`uvicorn app.main:app`) — not hypothetical examples.

**Chat completion**, explicit model override:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Explain what a semantic cache does in one sentence."}]}'
```
```json
{"id":"chatcmpl-ae3f5a06fc20","object":"chat.completion","created":1790337010,"model":"mock-gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"[MockProvider (mock-gpt-4o)] Responded to: 'Explain what a semantic cache does in one sentence.'"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":13,"total_tokens":22}}
```

**Streaming** (`"stream": true`) — real Server-Sent Events, one incremental chunk per event:

```
data: {"id":"chatcmpl-0d5f3829de84","object":"chat.completion.chunk","model":"mock-gpt-4o","choices":[{"index":0,"delta":{"content":"[MockProvider"},"finish_reason":null}],"usage":null}

data: {"id":"chatcmpl-f2d38da1e3da","object":"chat.completion.chunk","model":"mock-gpt-4o","choices":[{"index":0,"delta":{"content":" (mock-gpt-4o)]"},"finish_reason":null}],"usage":null}

... (more chunks) ...

data: [DONE]
```

**Validation error** — malformed requests are rejected before touching a model:

```bash
curl http://localhost:8000/v1/chat/completions -d '{"model": "mock-gpt-4o", "messages": []}'
```
```json
{"error":{"message":"Invalid request payload","type":"validation_error","details":[{"type":"too_short","loc":["body","messages"],"msg":"List should have at least 1 item after validation, not 0"}],"request_id":"7e631854-a264-4bbe-b2e0-03160617e3b7"}}
```
`HTTP 422`

**Graceful degradation, captured live**: while gathering these examples, the real `OPENAI_API_KEY` configured for this run had exhausted its OpenAI credits. Ingesting a document for RAG (which needs a real embedding call) hit that failure — and the gateway didn't crash, it logged a warning and returned a clean response with zero chunks stored, exactly per the degrade-to-miss design in `app/cache/semantic_cache.py` and `app/rag/document_store.py`:

```bash
curl http://localhost:8000/v1/documents -d '{"text": "Our refund policy allows returns within 30 days."}'
```
```json
{"document_id":"95babdffdff14281a109497d5172f650","chunks_created":0}
```
`HTTP 201` — the server log for this request: `WARNING: Semantic cache store skipped, embedding failed: Provider 'openai-embeddings' returned error: HTTP 429: insufficient_quota`

**Metrics** (`GET /metrics`, Prometheus text format) after a few requests:

```
gateway_requests_total 2
gateway_cache_hits_total 0
gateway_cache_misses_total 2
gateway_cost_usd_total 0.000218
gateway_latency_ms_avg 2.8
gateway_requests_by_model_total{model="mock-gpt-4o"} 2
```

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
- [x] **Milestone 18 — Automated Testing & CI/CD**
- [x] **Milestone 19 — GCP / Remote GPU Deployment** (scripts written & documented, not live-deployed — no GCP billing account in this environment)
- [ ] Milestone 20 — Optional A2A Interoperability
- [x] **Milestone 21 — Final Documentation & Demo**

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

# Copy the example env file and fill in your own values (OPENAI_API_KEY, etc.)
cp .env.example .env
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

### Running with Docker Compose

`docker-compose.yml` brings up the gateway (built from `Dockerfile`) plus a `pgvector/pgvector` Postgres container:

```bash
docker compose up --build
```

The Postgres container isn't wired into the app yet — the semantic cache, RAG document store, and MCP PostgreSQL server all still use their in-memory/SQLite stand-ins (each marked with a `ponytail: swap for pgvector` comment). It's there so the infrastructure exists ahead of that swap. `docker compose down -v` tears everything down including the named volumes.

### Deploying to GCP (Optional)

Scripts in `deploy/gcp/` document (but don't automatically run) a two-piece cloud deployment: `deploy-gateway-cloudrun.sh` builds and deploys the CPU-only gateway to Cloud Run (serverless, pay-per-request), and `deploy-vllm-gpu-vm.sh` provisions a real GPU-backed VM running vLLM — the genuine version of what `VLLM_SIMULATE_LOCAL` fakes locally on hardware without enough VRAM.

**These are not free** — a T4 GPU VM runs roughly $0.35–$0.55/hr in `us-central1` while it's up; Cloud Run has a free tier but bills per request beyond it. Both scripts print the teardown command to stop billing once you're done. Requires the `gcloud` CLI authenticated against a GCP project with billing enabled:

```bash
export GCP_PROJECT_ID=your-project-id
./deploy/gcp/deploy-gateway-cloudrun.sh   # CPU-only gateway → Cloud Run
./deploy/gcp/deploy-vllm-gpu-vm.sh        # real vLLM on a GPU VM (optional)
```

### Running Tests

```bash
pytest
```

### Continuous Integration

`.github/workflows/ci.yml` runs on every push/PR to `main`: a `test` job (installs `requirements.txt`, runs the full `pytest` suite) and a `docker-build` job (builds the image from the `Dockerfile`). The latter also gives automated verification of the Docker setup on every push, independent of whether Docker is running on any given contributor's machine locally.

## Known Limitations & Design Tradeoffs

Being upfront about what's real versus simulated, and what's genuinely finished versus written-but-unverified:

- **vLLM is simulated by default** (`VLLM_SIMULATE_LOCAL=true`) — this project was built on a GTX 1650 (4GB VRAM), not enough to run a real model locally. `vllm_provider.py` still implements a real HTTP/SSE client for an actual vLLM server; `deploy/gcp/deploy-vllm-gpu-vm.sh` provisions genuine GPU hardware to remove the simulation, but that script has not been executed in this environment (no GCP billing account available here).
- **Docker Compose is written and syntax-validated, not build/run-verified** — the Docker daemon wasn't available in this development environment. The `docker-build` CI job (Milestone 18) now builds it on every push, which is real, automated verification going forward — but it hasn't been run interactively end-to-end (`docker compose up` + a live curl) by the person who wrote it.
- **The semantic cache, RAG document store, and MCP PostgreSQL server are in-memory/SQLite stand-ins**, not the real pgvector/Postgres backend the architecture diagram shows. Each is marked with a `ponytail: swap at Milestone 17` comment and shares the same interface a real swap would need — the Postgres+pgvector container in `docker-compose.yml` exists, but nothing in the app code talks to it yet.
- **The example OpenAI-dependent output in the Demo section above shows a real failure, not a real success** — the API key used had no remaining credits when these examples were captured, so the semantic cache/RAG embedding calls degrade to a graceful miss rather than demonstrating a cache hit or real retrieval. The code path for a real hit is exercised by the test suite (with a fake embedding client), just not by this live capture.
