from typing import Any, Dict, Optional
import os

from fastapi import Header, HTTPException, status


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean-like environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# PUBLIC_INTERFACE
def auth_required(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Validate Authorization: Bearer <token> against API_TOKEN env var.

    Behavior:
      - If env ALLOW_NO_AUTH is true, requests without Authorization are allowed and
        an anonymous context is returned.
      - Otherwise, requires 'Authorization: Bearer <token>' where <token> matches API_TOKEN.
      - If the header is malformed or the token mismatches, raise 401 with WWW-Authenticate.

    Environment variables (must be set via .env, do not hardcode values):
      - API_TOKEN: Expected bearer token for simple auth.
      - ALLOW_NO_AUTH: If true, allows missing/empty Authorization header.

    Returns:
      Dict user context with keys: user_id, scopes, authenticated.
    """
    allow_no_auth = _env_bool("ALLOW_NO_AUTH", default=False)
    expected = os.getenv("API_TOKEN")

    # Allow no auth if configured (anonymous access)
    if (authorization is None or not authorization.strip()) and allow_no_auth:
        return {"user_id": "anonymous", "scopes": [], "authenticated": False}

    # Otherwise, require proper Bearer token
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()

    if expected is None or token != expected:
        # Either no configured API_TOKEN or mismatch
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Minimal user context; could be extended later
    return {"user_id": "api-token-user", "scopes": ["inference:run"], "authenticated": True}
