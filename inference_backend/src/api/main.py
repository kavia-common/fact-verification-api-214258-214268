from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Keep CORS setup minimal as requested
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # to be restricted in later steps/environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint remains unchanged for compatibility
@app.get("/", tags=["health"], summary="Health Check")
def health_check():
    """Simple health check endpoint."""
    return {"message": "Healthy"}


# Router registration (kept minimal; routes are placeholders for now)
try:
    from src.api.routes.inference import router as inference_router
    app.include_router(inference_router, prefix="/inference", tags=["inference"])
except Exception:
    # In case scaffold modules are missing during intermediate states,
    # avoid breaking the health endpoint or CORS configuration.
    pass
