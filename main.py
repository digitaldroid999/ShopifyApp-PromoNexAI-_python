"""
PromoNex Python API Service — entry point.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Or:
    python -m uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("Starting PromoNex API service")
    yield
    logger.info("Shutting down PromoNex API service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="PromoNex API",
        description="Request/response API server for PromoNex.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        """Log 422 validation errors so server logs show missing/invalid fields."""
        logger.warning("Request validation failed (422) path=%s detail=%s", request.url.path, exc.errors())
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.get("/health", tags=["health"])
    async def health():
        """Health check for load balancers and monitoring."""
        return {"status": "ok", "service": "promonex-api"}

    # Remotion bridge (Next.js <-> Remotion server)
    from app.api.remotion_routes import router as remotion_router
    app.include_router(remotion_router)

    # Image composite: save to Shopify app public folder, return /composited_images/{name}
    from app.api.composite_routes import router as composite_router
    app.include_router(composite_router)

    # Audio: script generation and audio generation (save to public/generated_audio/{user_id}/{short_id}/)
    from app.api.audio_routes import router as audio_router
    app.include_router(audio_router)

    # Background: async generation (OpenAI + Vertex Imagen) and extract-prompt
    from app.api.background_routes import router as background_router
    app.include_router(background_router)

    # Merge: finalize short (merge scenes + audio, upload final video)
    from app.api.merge_routes import router as merge_router
    app.include_router(merge_router)

    # Main API routes (scraping, video, merging, etc.) — enable after fixing missing services
    # from app.api.routes import router as main_router
    # app.include_router(main_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
