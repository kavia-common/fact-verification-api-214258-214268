from typing import Any, Dict


# PUBLIC_INTERFACE
def auth_required() -> Dict[str, Any]:
    """Placeholder authentication dependency.

    Note:
      - In a future step, validate tokens/headers and attach user context.
      - Do not read environment variables directly here; rely on settings module in future steps.

    Returns:
      A minimal user context dictionary for scaffolding.
    """
    # Placeholder user context; replace with real auth validation later
    return {"user_id": "anonymous", "scopes": [], "authenticated": True}
