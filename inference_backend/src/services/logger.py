from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _default_level() -> int:
    """Resolve default log level from environment later if needed."""
    # Keep it simple for now; FastAPI/uvicorn often controls handlers.
    return logging.INFO


def _ensure_logger() -> logging.Logger:
    """Create a root application logger configured for JSON-like structured logs."""
    logger = logging.getLogger("inference")
    if not logger.handlers:
        logger.setLevel(_default_level())
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(_default_level())

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                base: Dict[str, Any] = {
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "time": int(time.time() * 1000),
                    "logger": record.name,
                }
                # Attach extra dict if provided
                extra_dict = getattr(record, "extra_dict", None)
                if isinstance(extra_dict, dict):
                    base.update(extra_dict)
                return json.dumps(base, ensure_ascii=False)

        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


_LOGGER = _ensure_logger()


def log_info(message: str, **fields: Any) -> None:
    """Log an info message with structured fields merged into the record."""
    _LOGGER.info(message, extra={"extra_dict": fields})


def log_error(message: str, **fields: Any) -> None:
    """Log an error message with structured fields."""
    _LOGGER.error(message, extra={"extra_dict": fields})


def log_debug(message: str, **fields: Any) -> None:
    """Log a debug message with structured fields."""
    _LOGGER.debug(message, extra={"extra_dict": fields})


@contextmanager
def time_block(
    name: str,
    *,
    request_id: Optional[str] = None,
    client_label: Optional[str] = None,
    **fields: Any,
):
    """Context manager that logs start and end with elapsed time in ms.

    Parameters:
      name: Logical operation name (e.g., 'claim_search', 'claim_score').
      request_id: X-Request-ID for correlation.
      client_label: Optional client label.
      fields: Additional fields to include (e.g., sentence_index, claim_preview).
    """
    start = time.perf_counter()
    log_info(
        f"{name}:start",
        event="timing_start",
        op=name,
        request_id=request_id,
        client_label=client_label,
        **fields,
    )
    try:
        yield
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - start) * 1000.0)
        log_error(
            f"{name}:error",
            event="timing_error",
            op=name,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
            client_label=client_label,
            error=str(exc),
            **fields,
        )
        raise
    else:
        elapsed_ms = int((time.perf_counter() - start) * 1000.0)
        log_info(
            f"{name}:end",
            event="timing_end",
            op=name,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
            client_label=client_label,
            **fields,
        )
