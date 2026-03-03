"""
Shared security utilities — rate limiter, security headers middleware.

This module is kept dependency-free of the app routers so it can be imported
by both ``src.main`` and individual route modules without circular imports.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.config import settings


# =============================================================================
# Rate Limiter (slowapi) — shared singleton
# =============================================================================

limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # HSTS — only enforce when the deployment is behind HTTPS
        if not settings.is_development:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
        return response
