import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.config import settings
from src.history.routes import history_router
from src.rag.routes import rag_router
from src.schemas import APIInfoResponse, HealthCheckResponse
from src.security import SecurityHeadersMiddleware, limiter


# =============================================================================
# Logging
# =============================================================================

def setup_logging():
    """Configure logging to save to timestamped files."""
    log_dir = Path("logs/server_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


logger = setup_logging()


# =============================================================================
# App Factory
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Capstone API server started successfully")
    logger.info("📚 RAG-based chat system for Medical Misinformation Detection")
    yield
    logger.info("🛑 Capstone API server shutting down")


# Conditionally disable interactive docs in production
_docs_url = "/docs" if settings.is_development else None
_redoc_url = "/redoc" if settings.is_development else None

app = FastAPI(
    title="Capstone API",
    description="A RAG-based chat system for Medical Misinformation Detection",
    version="1.0.0",
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    lifespan=lifespan,
)

# Attach rate limiter to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# CORS — tightened to known origins
# =============================================================================

_cors_origins: list[str] = settings.cors_origin_list

if not _cors_origins:
    if settings.is_development:
        # Wide-open during local development only
        _cors_origins = ["*"]
        logger.warning("CORS: allowing ALL origins (development mode)")
    else:
        # Production: same-origin only (empty list = no extra origins allowed)
        _cors_origins = []
        logger.info("CORS: same-origin only (no extra origins configured)")
else:
    logger.info(f"CORS: allowing origins {_cors_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^chrome-extension://.*",  # Allow all chrome extensions (works for any unpacked extension ID)
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-OpenRouter-API-Key"],
)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)


# =============================================================================
# Request Logging Middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses."""
    start_time = datetime.now()

    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )

    response = await call_next(request)

    process_time = (datetime.now() - start_time).total_seconds()

    logger.info(
        f"Response: {response.status_code} for {request.method} {request.url.path} "
        f"- {process_time:.3f}s"
    )

    return response


# =============================================================================
# Routers
# =============================================================================

app.include_router(history_router)
app.include_router(rag_router)


# =============================================================================
# API Endpoints (must be registered BEFORE the SPA catch-all route)
# =============================================================================

@app.get("/", response_model=APIInfoResponse)
async def root() -> APIInfoResponse:
    """Root endpoint providing API information."""
    return APIInfoResponse(
        message="Welcome to Capstone API",
        description="A RAG-based chat system for Medical Misinformation Detection",
        version="1.0.0",
        endpoints={
            "health": "/health",
            "rag": "/rag",
            "history": "/history",
        },
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint for monitoring."""
    return HealthCheckResponse(
        status="healthy", service="Capstone API", version="1.0.0"
    )


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(500)
async def internal_server_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle internal server errors with custom response."""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": str(exc) if settings.is_development else None,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle 404 errors with custom response."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found.",
            "path": str(request.url.path),
        },
    )


# =============================================================================
# Frontend SPA Routing (must be LAST - catch-all route)
# =============================================================================

frontend_dist_path = Path("frontend/dist")
if frontend_dist_path.exists() and frontend_dist_path.is_dir():
    app.mount(
        "/assets", StaticFiles(directory=frontend_dist_path / "assets"), name="assets"
    )
    logger.info("Frontend assets mounted at /assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA for all non-API routes."""
        if (
            full_path.startswith("api/")
            or full_path.startswith("docs")
            or full_path.startswith("redoc")
            or full_path.startswith("files/")
        ):
            raise HTTPException(status_code=404, detail="Not found")

        file_path = frontend_dist_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        index_file = frontend_dist_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend not found")

    logger.info("SPA routing configured for frontend")
else:
    logger.warning("Frontend dist directory not found - frontend will not be served")
