import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.inference import router as inference_router


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean-like env values safely; never raise on import."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    """Parse comma-separated env list safely, trimming whitespace and ignoring empties."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return list(default)
        items = [x.strip() for x in str(raw).split(",")]
        return [x for x in items if x]
    except Exception:
        return list(default)


# Minimal FastAPI app with metadata and tags for future expansion
openapi_tags = [
    {
        "name": "health",
        "description": "Service health and diagnostics endpoints.",
    },
    {
        "name": "inference",
        "description": "Endpoints for fact verification inference (claim detection, search, scoring, streaming).",
    },
]

app = FastAPI(
    title="Fact Verification Inference API",
    description="FastAPI backend for sentence splitting, claim detection, web search, evidence scoring, and streaming results.",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# Parameterized CORS configuration via environment variables.
# Note: all values should be supplied via .env; no hard-coded defaults beyond safe dev values.
# - CORS_ALLOW_ORIGINS: comma-separated list (e.g., "https://app.com,https://staging.app.com"). Default ["*"] for dev.
# - CORS_ALLOW_CREDENTIALS: boolean ("true"/"false"), default True.
# - CORS_ALLOW_METHODS / CORS_ALLOW_HEADERS: comma-separated lists, default ["*"].
cors_allow_origins = _env_list("CORS_ALLOW_ORIGINS", default=["*"])
cors_allow_credentials = _env_bool("CORS_ALLOW_CREDENTIALS", default=True)
cors_allow_methods = _env_list("CORS_ALLOW_METHODS", default=["*"])
cors_allow_headers = _env_list("CORS_ALLOW_HEADERS", default=["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=cors_allow_methods,
    allow_headers=cors_allow_headers,
)

# Startup environment validation is intentionally disabled to allow the app
# to run even when API_TOKEN is missing and ALLOW_NO_AUTH is false.


# PUBLIC_INTERFACE
@app.get(
    "/health",
    tags=["health"],
    summary="Health Check",
    description="Lightweight unauthenticated health check. Returns a minimal JSON payload and 200 OK.",
    responses={
        200: {
            "description": "Service is up",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "app": "fact-verification-inference", "version": "0.1.0", "allow_no_auth": True}
                }
            },
        }
    },
)
def health_check():
    """Return 200 OK with minimal diagnostics to verify service status.

    Returns:
      JSON object like:
        {
          "status": "ok",
          "app": "fact-verification-inference",
          "version": "<app version>",
          "allow_no_auth": <bool from env>
        }
    Notes:
      - This endpoint is intentionally unauthenticated and performs no external calls.
      - It should remain extremely fast and dependency-free.
    """
    return {
        "status": "ok",
        "app": "fact-verification-inference",
        "version": app.version if hasattr(app, "version") else "unknown",
        "allow_no_auth": _env_bool("ALLOW_NO_AUTH", default=False),
    }


# Router registration for inference endpoints
app.include_router(inference_router, prefix="/inference", tags=["inference"])
