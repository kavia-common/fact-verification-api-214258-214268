from fastapi import APIRouter, Depends
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
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
) -> InferenceResponse:
    """Run inference placeholder. Returns an acknowledgement and echoes inputs.

    Parameters:
      - payload: JSON with input text and optional parameters (placeholder).
      - user: injected by auth dependency (placeholder).
      - label: injected by label dependency (placeholder).

    Returns:
      A simple JSON response indicating the request was received.
    """
    # We return a minimal InferenceResponse instance to match the schema while keeping placeholder behavior.
    return InferenceResponse(
        status="accepted",
        sentences=[],
        claims=[],
        metadata={"user": user, "label": label, "echo": payload.model_dump()},
        message="Inference scaffolding in place.",
    )


# PUBLIC_INTERFACE
@router.get(
    "/stream",
    summary="Stream inference results",
    description="Placeholder streaming endpoint to be implemented with chunked responses.",
)
def stream_inference(
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
) -> Dict[str, Any]:
    """Streaming placeholder. Will be replaced by Server-Sent Events or WebSocket.

    Parameters:
      - user: injected by auth dependency (placeholder).
      - label: injected by label dependency (placeholder).

    Returns:
      A simple JSON response indicating streaming will be implemented later.
    """
    return {
        "status": "accepted",
        "message": "Streaming scaffolding in place. Implementation in a later step.",
        "user": user,
        "label": label,
    }
