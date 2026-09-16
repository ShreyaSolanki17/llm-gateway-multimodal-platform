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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
