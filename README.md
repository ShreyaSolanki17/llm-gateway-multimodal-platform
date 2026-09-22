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

## Milestone Status

- [x] **Milestone 0 — Project Definition & Repository Setup**
- [x] **Milestone 1 — FastAPI LLM Gateway**
- [x] **Milestone 2 — Model Provider Abstraction**
- [x] **Milestone 3 — Model Router**
- [x] **Milestone 4 — vLLM Model Serving**
- [x] **Milestone 5 — Token, Latency & Cost Tracking**
- [x] **Milestone 6 — Semantic Cache**
- [x] **Milestone 7 — Multimodal Input Pipeline**
- [ ] Milestone 8 — RAG Pipeline
- [ ] Milestone 9 — Advanced Retrieval
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
