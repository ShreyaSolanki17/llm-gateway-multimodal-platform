from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.core.logging import logger
from app.core.middleware import RequestIDMiddleware
from app.core.exceptions import (
    GatewayException,
    gateway_exception_handler,
    validation_exception_handler,
)
from app.api.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} in [{settings.ENVIRONMENT}] mode...")
    if not settings.GATEWAY_API_KEY:
        logger.warning("GATEWAY_API_KEY is not configured -- authentication is DISABLED. Set it before exposing this gateway.")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-oriented LLM Gateway & Multimodal Inference Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Add Middleware
app.add_middleware(RequestIDMiddleware)

# Add Exception Handlers
app.add_exception_handler(GatewayException, gateway_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Include Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(documents_router)


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
        "health": "/health",
        "chat": "/v1/chat/completions",
    }
