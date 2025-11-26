from typing import Any, Dict, Optional
import uuid

from fastapi import Header, Request, Response


def _ensure_request_id(incoming: Optional[str]) -> str:
    """Return a valid request id: re-use incoming if present, otherwise generate."""
    rid = (incoming or "").strip()
    if not rid:
        return str(uuid.uuid4())
    return rid


# PUBLIC_INTERFACE
def label_context(
    request: Request,
    response: Response,
    x_client_label: Optional[str] = Header(default=None, alias="X-Client-Label"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
) -> Dict[str, Any]:
    """Capture client label and request id, attach to request.state and response headers.

    Headers:
      - X-Client-Label: Optional free-form client/application label.
      - X-Request-ID: Optional request id supplied by the client; generated if missing.

    Side-effects:
      - Sets request.state.client_label and request.state.request_id for downstream usage.
      - Adds X-Client-Label and X-Request-ID to the response headers for traceability.

    Returns:
      Dict label context with keys: request_id, client_label.
    """
    req_id = _ensure_request_id(x_request_id)
    client_label = (x_client_label or "").strip() or None

    # Attach to request context for deeper services to use if needed
    request.state.request_id = req_id
    request.state.client_label = client_label

    # Echo/propagate on response
    response.headers["X-Request-ID"] = req_id
    if client_label:
        response.headers["X-Client-Label"] = client_label

    return {"request_id": req_id, "client_label": client_label}
