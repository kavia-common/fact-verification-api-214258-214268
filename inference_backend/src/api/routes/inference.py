from fastapi import APIRouter, Depends
from typing import Any, Dict

from src.api.deps.auth import auth_required
from src.api.deps.labels import label_context

router = APIRouter()


# PUBLIC_INTERFACE
@router.post(
    "/run",
    summary="Run inference for a provided text",
    description="Accepts text and initiates the fact verification pipeline. Placeholder implementation.",
)
def run_inference(
    payload: Dict[str, Any],
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
) -> Dict[str, Any]:
    """Run inference placeholder. Returns an acknowledgement and echoes inputs.

    Parameters:
      - payload: JSON with input text and optional parameters (placeholder).
      - user: injected by auth dependency (placeholder).
      - label: injected by label dependency (placeholder).

    Returns:
      A simple JSON response indicating the request was received.
    """
    return {
        "status": "accepted",
        "message": "Inference scaffolding in place.",
        "echo": payload,
        "user": user,
        "label": label,
    }


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
