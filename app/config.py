from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application Settings loaded from environment variables or .env file."""

    APP_NAME: str = "LLM Gateway & Multimodal Platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    LOG_LEVEL: str = "INFO"

    # vLLM Configuration
    VLLM_SERVER_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
    VLLM_TIMEOUT: float = 30.0
    VLLM_ENABLED: bool = True
    VLLM_SIMULATE_LOCAL: bool = True  # Handles local low VRAM (GTX 1650) development

    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None

    # Semantic Cache Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # RAG Pipeline Configuration
    RAG_CHUNK_SIZE: int = 200
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 3
    RAG_MIN_SIMILARITY: float = 0.3

    # PostgreSQL MCP Server Configuration (SQLite-backed for now; swaps to
    # real Postgres at Milestone 17 once Docker infra exists)
    MCP_DB_PATH: str = "./data/mcp_gateway.db"
    MCP_QUERY_ROW_LIMIT: int = 100

    # Security & Guardrails Configuration
    GATEWAY_API_KEY: Optional[str] = None  # unset = auth disabled (dev convenience; logs a startup warning)
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    MAX_MESSAGES_PER_REQUEST: int = 50
    MAX_TOTAL_CONTENT_CHARS: int = 50000
    MAX_DOCUMENT_CHARS: int = 200000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
