import json
import logging
import sys
from app.config import settings


class JSONFormatter(logging.Formatter):
    """Emits each log record as a single JSON line, so log-aggregation tooling
    (Datadog, ELK, CloudWatch, Loki, ...) can parse fields without a custom grok pattern."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    """Configure structured JSON console logging for the application."""
    logger = logging.getLogger("llm_gateway")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


logger = setup_logging()
