import os
from typing import List, Optional

from src.services.logger import log_info


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment variable values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    """Parse a comma-separated list environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return list(default or [])
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _effective_search_provider() -> str:
    """Return the requested search provider (normalized lowercase) or 'auto'."""
    return (os.getenv("SEARCH_PROVIDER") or "auto").strip().lower()


def _has_bing_key() -> bool:
    """Detect if a Bing-compatible key is present in env."""
    return bool((os.getenv("BING_API_KEY") or os.getenv("SEARCH_API_KEY") or "").strip())


def _should_require_bing_key(provider: str) -> bool:
    """
    Determine whether a Bing key should be required based on SEARCH_PROVIDER.

    Rules:
      - If SEARCH_PROVIDER == 'bing' -> require key.
      - If SEARCH_PROVIDER == 'auto' -> do NOT require a key (we fall back to keyless providers);
        but if someone intends to use Bing under 'auto' and set a key, that's fine. We only fail fast
        when explicitly set to 'bing' without a key.
      - For 'duckduckgo' or 'wikipedia' -> never require Bing key.
    """
    if provider == "bing":
        return True
    return False


# PUBLIC_INTERFACE
def validate_environment() -> None:
    """
    Validate critical environment settings.

    IMPORTANT:
    This function no longer raises at import or startup. It is provided for
    manual diagnostics only and must be invoked explicitly by an operator or a
    management script. Importing this module will not perform any validation.

    The historical checks were:
      - API token/auth mode consistency
      - Search provider API key presence for 'bing'
      - CORS configuration logging

    To preserve zero side effects on import, this function is currently a no-op
    unless called directly, in which case it will only log the effective config.
    """
    allow_no_auth = _env_bool("ALLOW_NO_AUTH", default=False)
    provider = _effective_search_provider()

    cors_allow_origins = _env_list("CORS_ALLOW_ORIGINS", default=["*"])
    cors_allow_credentials = _env_bool("CORS_ALLOW_CREDENTIALS", default=True)
    cors_allow_methods = _env_list("CORS_ALLOW_METHODS", default=["*"])
    cors_allow_headers = _env_list("CORS_ALLOW_HEADERS", default=["*"])

    # Only log; do not raise
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
        note="validation disabled at startup",
    )


if __name__ == "__main__":
    # Allow running this module as a script for manual diagnostics
    validate_environment()
