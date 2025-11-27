import os
from typing import List, Optional

# NOTE:
# This module is intentionally side-effect free at import time.
# Do NOT perform any environment validation, logging, or raises at module import.
# Only define helpers and callable functions. Execution is guarded by __main__.

try:
    # Optional import: logging helper. If unavailable for any reason during tooling,
    # skip logging to preserve zero side effects and import safety.
    from src.services.logger import log_info  # type: ignore
except Exception:  # pragma: no cover - defensive fallback to avoid import-time failures
    def log_info(message: str, **fields):  # type: ignore
        return  # no-op fallback


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment variable values safely without raising on import."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return default
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        # Never raise at import or call-sites; fall back to default
        return default


def _env_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    """Parse a comma-separated list environment variable safely."""
    try:
        raw = os.getenv(name)
        if raw is None:
            return list(default or [])
        parts = [p.strip() for p in str(raw).split(",")]
        return [p for p in parts if p]
    except Exception:
        return list(default or [])


def _effective_search_provider() -> str:
    """Return the requested search provider (normalized lowercase) or 'auto'."""
    try:
        return (os.getenv("SEARCH_PROVIDER") or "auto").strip().lower()
    except Exception:
        return "auto"


def _has_bing_key() -> bool:
    """Detect if a Bing-compatible key is present in env."""
    try:
        return bool((os.getenv("BING_API_KEY") or os.getenv("SEARCH_API_KEY") or "").strip())
    except Exception:
        return False


def _should_require_bing_key(provider: str) -> bool:
    """
    Determine whether a Bing key should be required based on SEARCH_PROVIDER.

    Rules:
      - If SEARCH_PROVIDER == 'bing' -> require key.
      - Otherwise -> do not require a key (fallback providers exist).
    """
    return provider == "bing"


# PUBLIC_INTERFACE
def validate_environment() -> None:
    """
    Manually log effective environment configuration for diagnostics.

    IMPORTANT:
    - This function does not raise; it only logs a snapshot of effective settings.
    - Importing this module performs no validation and has zero side effects.

    Logged details:
      - CORS settings
      - Search provider selection and whether a Bing key is present
      - Auth mode (anonymous allowed vs token required)
    """
    allow_no_auth = _env_bool("ALLOW_NO_AUTH", default=False)
    provider = _effective_search_provider()

    cors_allow_origins = _env_list("CORS_ALLOW_ORIGINS", default=["*"])
    cors_allow_credentials = _env_bool("CORS_ALLOW_CREDENTIALS", default=True)
    cors_allow_methods = _env_list("CORS_ALLOW_METHODS", default=["*"])
    cors_allow_headers = _env_list("CORS_ALLOW_HEADERS", default=["*"])

    # Only log; never raise
    log_info(
        "startup.cors_config",
        event="startup_config",
        cors_allow_origins=cors_allow_origins,
        cors_allow_credentials=cors_allow_credentials,
        cors_allow_methods=cors_allow_methods,
        cors_allow_headers=cors_allow_headers,
        search_provider=provider,
        auth_mode=("anonymous_allowed" if allow_no_auth else "token_required"),
        has_bing_key=_has_bing_key(),
        note="validation disabled at startup; call validate_environment() manually if needed",
    )


if __name__ == "__main__":
    # Allow running this module as a script for manual diagnostics only.
    # Importing this module elsewhere will not execute validation.
    validate_environment()
