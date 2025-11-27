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
    Validate critical environment settings at application startup.

    This function performs:
      1) Authentication checks:
         - If ALLOW_NO_AUTH is false (default) and API_TOKEN is missing, raise descriptive RuntimeError.
      2) Search configuration checks:
         - If SEARCH_PROVIDER is 'bing' and neither BING_API_KEY nor SEARCH_API_KEY is set, raise RuntimeError.
         - Do not force Bing key for 'auto', 'duckduckgo', or 'wikipedia' providers.
      3) Log the effective CORS configuration.

    Environment variables referenced (must be provided via .env by the orchestrator):
      - ALLOW_NO_AUTH: "true"/"false" (default false)
      - API_TOKEN: required when ALLOW_NO_AUTH is false
      - SEARCH_PROVIDER: 'auto' | 'bing' | 'duckduckgo' | 'wikipedia' (default 'auto')
      - BING_API_KEY / SEARCH_API_KEY: required when SEARCH_PROVIDER == 'bing'
      - CORS_ALLOW_ORIGINS, CORS_ALLOW_CREDENTIALS, CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS

    Raises:
      RuntimeError with a helpful message if configuration is invalid.
    """
    # 1) Auth validation
    allow_no_auth = _env_bool("ALLOW_NO_AUTH", default=False)
    api_token = os.getenv("API_TOKEN")

    if not allow_no_auth and not (api_token and api_token.strip()):
        raise RuntimeError(
            "Configuration error: API_TOKEN is required but missing while ALLOW_NO_AUTH is false.\n"
            "To fix: set API_TOKEN=<your_token> in the environment (preferred), or set ALLOW_NO_AUTH=true\n"
            "for anonymous development access (NOT recommended for production)."
        )

    # 2) Search provider validation
    provider = _effective_search_provider()
    require_bing = _should_require_bing_key(provider)
    if require_bing and not _has_bing_key():
        raise RuntimeError(
            "Configuration error: SEARCH_PROVIDER='bing' requires a Bing API key but none was found.\n"
            "To fix: set BING_API_KEY=<your_bing_key> (or SEARCH_API_KEY=<your_bing_key>) in the environment.\n"
            "Alternatively, choose a keyless provider by setting SEARCH_PROVIDER='duckduckgo' or 'wikipedia',\n"
            "or set SEARCH_PROVIDER='auto' to fall back to keyless providers when no key is provided."
        )

    # 3) Log CORS configuration to aid diagnostics
    cors_allow_origins = _env_list("CORS_ALLOW_ORIGINS", default=["*"])
    cors_allow_credentials = _env_bool("CORS_ALLOW_CREDENTIALS", default=True)
    cors_allow_methods = _env_list("CORS_ALLOW_METHODS", default=["*"])
    cors_allow_headers = _env_list("CORS_ALLOW_HEADERS", default=["*"])

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
    )
