import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.inference import router as inference_router


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean-like env values safely."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    """Parse comma-separated env list, trimming whitespace, ignoring empties."""
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [x.strip() for x in raw.split(",")]
    return [x for x in items if x]


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

# Parameterized CORS configuration via environment variables
# CORS_ALLOW_ORIGINS: comma-separated list. Defaults to "*" (public) for development.
# CORS_ALLOW_CREDENTIALS: boolean ("true"/"false"), default True
# CORS_ALLOW_METHODS / CORS_ALLOW_HEADERS: comma-separated lists, default "*"
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


# Health endpoint remains unchanged for compatibility
@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Simple health check endpoint."""
    return {"message": "Healthy"}


# Router registration for inference endpoints
app.include_router(inference_router, prefix="/inference", tags=["inference"])
