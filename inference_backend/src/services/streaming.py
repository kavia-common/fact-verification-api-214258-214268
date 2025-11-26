from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Iterator, Optional, Union, Callable

try:
    # FastAPI/Starlette may not be imported where this module is used for non-HTTP contexts.
    from fastapi.responses import StreamingResponse
except Exception:  # pragma: no cover - optional import
    StreamingResponse = None  # type: ignore[assignment]


# Internal constants for event types to keep them consistent across helpers.
_EVENT_SENTENCE = "sentence"
_EVENT_EVIDENCE = "evidence"
_EVENT_SCORE = "score"
_EVENT_DONE = "done"
_EVENT_ERROR = "error"


def _ndjson_line(obj: Dict[str, Any]) -> str:
    """Serialize a dict to a JSON line with newline terminator."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"


# PUBLIC_INTERFACE
def make_sentence_event(sentence: Dict[str, Any], seq: int, progress: Optional[float] = None) -> Dict[str, Any]:
    """Create a sentence event payload suitable for streaming.

    Parameters:
      sentence: A dict compatible with SentenceChunk schema.
      seq: Sequence number for the event.
      progress: Optional normalized progress [0, 1].

    Returns:
      Event dict with fields: event, data, seq, progress.
    """
    evt: Dict[str, Any] = {"event": _EVENT_SENTENCE, "data": sentence, "seq": int(seq)}
    if progress is not None:
        evt["progress"] = float(progress)
    return evt


# PUBLIC_INTERFACE
def make_evidence_event(evidence: Dict[str, Any], seq: int, progress: Optional[float] = None) -> Dict[str, Any]:
    """Create an evidence event payload for a claim.

    Parameters:
      evidence: Evidence payload (e.g., {'claim': str, 'supporting_evidence': [...], 'refuting_evidence': [...]}).
      seq: Sequence number.
      progress: Optional progress in [0, 1].

    Returns:
      Event dict.
    """
    evt: Dict[str, Any] = {"event": _EVENT_EVIDENCE, "data": evidence, "seq": int(seq)}
    if progress is not None:
        evt["progress"] = float(progress)
    return evt


# PUBLIC_INTERFACE
def make_score_event(score: Dict[str, Any], seq: int, progress: Optional[float] = None) -> Dict[str, Any]:
    """Create a score event payload for an evaluated claim.

    Parameters:
      score: Dict with aggregate scoring and label for the claim.
      seq: Sequence number.
      progress: Optional progress.

    Returns:
      Event dict.
    """
    evt: Dict[str, Any] = {"event": _EVENT_SCORE, "data": score, "seq": int(seq)}
    if progress is not None:
        evt["progress"] = float(progress)
    return evt


# PUBLIC_INTERFACE
def make_done_event(summary: Optional[Dict[str, Any]], seq: int = 0) -> Dict[str, Any]:
    """Create a done/completion event.

    Parameters:
      summary: Optional summary payload, e.g., overall InferenceResponse.
      seq: Sequence number (defaults to 0 if unused).

    Returns:
      Event dict.
    """
    return {"event": _EVENT_DONE, "data": (summary or {}), "seq": int(seq)}


# PUBLIC_INTERFACE
def make_error_event(message: Union[str, Dict[str, Any]], seq: int = 0) -> Dict[str, Any]:
    """Create an error event.

    Parameters:
      message: Error message string or dict payload describing the error.
      seq: Sequence number (defaults to 0 if unused).

    Returns:
      Event dict.
    """
    return {"event": _EVENT_ERROR, "data": {"message": message} if isinstance(message, str) else message, "seq": int(seq)}


# PUBLIC_INTERFACE
def ndjson_stream(
    items: Iterable[Dict[str, Any]],
    heartbeat: bool = False,
    heartbeat_interval_sec: float = 10.0,
    time_source: Callable[[], float] = time.time,
) -> Iterator[str]:
    """Yield NDJSON lines from an iterable of event dicts, with optional heartbeats.

    Each item yielded by the input iterable should be a dict constructed by the
    make_*_event helpers (sentence, evidence, score, done, error). This function
    converts them to newline-delimited JSON strings appropriate for streaming.

    Heartbeats:
      If heartbeat is True, this generator will periodically yield a heartbeat line
      (an empty object {}) to keep connections alive during long gaps. Many proxies
      and load balancers time out idle connections otherwise.

    Parameters:
      items: Iterable yielding event dictionaries.
      heartbeat: Whether to emit heartbeats.
      heartbeat_interval_sec: Minimum interval between heartbeats.
      time_source: Injectable time source primarily for tests.

    Yields:
      NDJSON lines as strings (including trailing '\n').
    """
    last_yield_ts = time_source()
    gap = max(0.5, float(heartbeat_interval_sec))

    def _maybe_heartbeat(now_ts: float) -> Optional[str]:
        nonlocal last_yield_ts
        if heartbeat and (now_ts - last_yield_ts) >= gap:
            last_yield_ts = now_ts
            # NDJSON heartbeat: an empty JSON object line
            return _ndjson_line({})
        return None

    # Iterate through items, yielding heartbeats when needed.
    for item in items:
        now = time_source()
        hb = _maybe_heartbeat(now)
        if hb is not None:
            yield hb
        yield _ndjson_line(item)
        last_yield_ts = now

    # After items complete, we don't emit extra heartbeats, but ensure final flush behavior
    # is up to the server/StreamingResponse.


# PUBLIC_INTERFACE
def stream_chunks(items: Iterable[Dict[str, Any]]) -> Iterable[str]:
    """Convert iterable of dict items into newline-delimited JSON strings.

    This is a minimal wrapper kept for backward compatibility. Prefer ndjson_stream
    for heartbeat support.

    Parameters:
      items: Iterable of dicts.

    Yields:
      NDJSON lines as strings.
    """
    for item in items:
        yield _ndjson_line(item)


# PUBLIC_INTERFACE
def as_fastapi_streaming_response(
    items: Iterable[Dict[str, Any]],
    media_type: str = "application/x-ndjson",
    heartbeat: bool = True,
    heartbeat_interval_sec: float = 10.0,
):
    """Build a FastAPI StreamingResponse configured for NDJSON event streaming.

    Parameters:
      items: Iterable of event dicts produced by make_*_event helpers.
      media_type: Response media type. Defaults to application/x-ndjson.
      heartbeat: Emit empty-object heartbeats periodically to keep the connection alive.
      heartbeat_interval_sec: Interval in seconds between heartbeats.

    Returns:
      StreamingResponse instance (if FastAPI is available) or raises RuntimeError otherwise.
    """
    if StreamingResponse is None:
        raise RuntimeError("FastAPI not available: cannot create StreamingResponse")

    generator = ndjson_stream(
        items=items,
        heartbeat=heartbeat,
        heartbeat_interval_sec=heartbeat_interval_sec,
    )
    return StreamingResponse(generator, media_type=media_type)


# PUBLIC_INTERFACE
def ndjson_error_response(message: str, status_code: int = 200):
    """Create a minimal NDJSON StreamingResponse that emits a single error event and completes.

    Parameters:
      message: Error message to include in the stream.
      status_code: HTTP status code to use (default 200 so clients can parse stream body).

    Returns:
      StreamingResponse emitting one error event line.
    """
    if StreamingResponse is None:
        raise RuntimeError("FastAPI not available: cannot create StreamingResponse")

    def _gen() -> Iterator[str]:
        yield _ndjson_line(make_error_event(message=message, seq=0))

    return StreamingResponse(_gen(), media_type="application/x-ndjson", status_code=status_code)
