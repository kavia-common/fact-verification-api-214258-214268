from fastapi import APIRouter, Depends, Response
from typing import Any, Dict

from src.api.deps.auth import auth_required
from src.api.deps.labels import label_context
from src.models.schemas import InferenceRequest, InferenceResponse  # type hints for OpenAPI

router = APIRouter()


# PUBLIC_INTERFACE
@router.post(
    "/run",
    summary="Run inference for a provided text",
    description="Accepts text and initiates the fact verification pipeline. Placeholder implementation.",
    response_model=InferenceResponse,
)
def run_inference(
    payload: InferenceRequest,
    response: Response,
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
) -> InferenceResponse:
    """Run inference placeholder. Returns an acknowledgement and echoes inputs.

    Parameters:
      - payload: JSON with input text and optional parameters (placeholder).
      - response: FastAPI Response to propagate headers set by dependencies.
      - user: injected by auth dependency.
      - label: injected by label dependency.

    Returns:
      A simple JSON response indicating the request was received.
    """
    # The label_context dependency already set X-Request-ID / X-Client-Label on response headers.
    meta = {
        "user": user,
        "label": label,  # contains request_id and client_label
        "echo": payload.model_dump(),
    }
    return InferenceResponse(
        status="accepted",
        sentences=[],
        claims=[],
        metadata=meta,
        message="Inference scaffolding in place.",
    )


# PUBLIC_INTERFACE
@router.get(
    "/stream",
    summary="Stream inference results",
    description="Placeholder streaming endpoint to be implemented with chunked responses.",
)
def stream_inference(
    response: Response,
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
) -> Dict[str, Any]:
    """Streaming placeholder. Will be replaced by Server-Sent Events or WebSocket.

    Parameters:
      - response: Response object so dependency header propagation is consistent.
      - user: injected by auth dependency.
      - label: injected by label dependency.

    Returns:
      A simple JSON response indicating streaming will be implemented later.
    """
    return {
        "status": "accepted",
        "message": "Streaming scaffolding in place. Implementation in a later step.",
        "user": user,
        "label": label,
    }
