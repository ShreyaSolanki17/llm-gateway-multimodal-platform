from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.core.logging import logger
from app.api.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} in [{settings.ENVIRONMENT}] mode...")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-oriented LLM Gateway & Multimodal Inference Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(health_router)


@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "docs": "/docs",
        "health": "/health",
    }
