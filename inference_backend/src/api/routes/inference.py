from fastapi import APIRouter, Depends, Response
from typing import Any, Dict, List, Iterator
import time

from src.api.deps.auth import auth_required
from src.api.deps.labels import label_context
from src.models.schemas import (
    InferenceRequest,
    InferenceResponse,
    SentenceChunk,
    ClaimResult,
)
from src.services.sentencizer import get_sentences
from src.services.claim_detection import detect_claims
from src.services.search import search_for_claim
from src.services.evidence_scoring import categorize_and_score
from src.services.streaming import (
    make_sentence_event,
    make_evidence_event,
    make_score_event,
    make_done_event,
    as_fastapi_streaming_response,
)
from src.services.logger import log_info, time_block  # structured logging

router = APIRouter()


# PUBLIC_INTERFACE
@router.post(
    "/run",
    summary="Run inference for a provided text",
    description="Runs the pipeline: sentencize → claim detection → per-claim search and scoring.",
    response_model=InferenceResponse,
)
def run_inference(
    payload: InferenceRequest,
    response: Response,
    # Removed auth dependency for this route to allow unauthenticated access
    label: Dict[str, Any] = Depends(label_context),
) -> InferenceResponse:
    """Run the full inference pipeline for the given input text.

    Pipeline:
      1) Sentencize the input text.
      2) Detect which sentences are claims.
      3) For each claim, perform web search and score evidence.
      4) Aggregate results into an InferenceResponse.

    Parameters:
      - payload: InferenceRequest containing text, top_k, language, streaming flag, and optional metadata.
      - response: FastAPI Response (used so label dependency can set headers).
      - user: Injected by auth_required dependency.
      - label: Injected by label_context dependency (request_id, client_label).

    Returns:
      InferenceResponse with detected sentences, claims, and processing metadata.
    """
    started = time.perf_counter()
    request_id = label.get("request_id")
    client_label = label.get("client_label")

    # Since auth is not required for this endpoint, provide a minimal anonymous user context for metadata.
    user: Dict[str, Any] = {"user_id": "anonymous", "scopes": [], "authenticated": False}

    # Respect options: derive limits with safe bounds.
    top_k = int(getattr(payload, "top_k", 5) or 5)
    # search_limit (number of results per claim) is mapped to top_k from request.
    search_limit = max(1, min(50, top_k))

    log_info(
        "inference.run:start",
        event="inference_start",
        request_id=request_id,
        client_label=client_label,
        top_k=top_k,
        search_limit=search_limit,
        language=payload.language,
        streaming=payload.streaming,
    )

    # 1) Sentence splitting
    with time_block("sentencize", request_id=request_id, client_label=client_label):
        raw_sentences: List[str] = get_sentences(payload.text or "")

    # 2) Claim detection
    with time_block("claim_detection", request_id=request_id, client_label=client_label):
        claim_flags: List[bool] = detect_claims(raw_sentences)

    # Build sentence chunks
    sentence_chunks: List[SentenceChunk] = []
    for s, is_claim in zip(raw_sentences, claim_flags):
        sentence_chunks.append(
            SentenceChunk(
                text=s,
                start_char=None,  # start/end offsets not tracked in our splitter
                end_char=None,
                is_claim=bool(is_claim),
            )
        )

    # 3) For each claim, perform search and 4) scoring
    claim_results: List[ClaimResult] = []
    for idx, (s, is_claim) in enumerate(zip(raw_sentences, claim_flags)):
        if not is_claim:
            continue

        claim_preview = s[:120]
        with time_block(
            "claim_search",
            request_id=request_id,
            client_label=client_label,
            sentence_index=idx,
            claim_preview=claim_preview,
        ):
            evidence = search_for_claim(s, top_k=search_limit)

        with time_block(
            "claim_score",
            request_id=request_id,
            client_label=client_label,
            sentence_index=idx,
            claim_preview=claim_preview,
        ):
            scored = categorize_and_score(claim=s, evidence=evidence, top_k=top_k)

        claim_results.append(
            ClaimResult(
                claim=s,
                sentence_index=idx,
                supporting_evidence=scored.get("supporting_evidence", []),
                refuting_evidence=scored.get("refuting_evidence", []),
                score=float(scored.get("score", 0.0)),
                label=scored.get("label"),
            )
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000.0)

    # Build metadata including processing time and context from dependencies.
    meta: Dict[str, Any] = {
        "processing_time_ms": elapsed_ms,
        "user": user,
        "label": label,  # contains request_id and client_label
        "request_metadata": payload.metadata or {},
        "options": {
            "top_k": top_k,
            "search_limit": search_limit,
            "language": payload.language,
            "streaming": payload.streaming,
        },
    }

    status = "completed"
    message = None

    log_info(
        "inference.run:end",
        event="inference_end",
        request_id=request_id,
        client_label=client_label,
        processing_time_ms=elapsed_ms,
        sentences=len(sentence_chunks),
        claims=len(claim_results),
    )

    return InferenceResponse(
        status=status,
        sentences=sentence_chunks,
        claims=claim_results,
        metadata=meta,
        message=message,
    )


# PUBLIC_INTERFACE
@router.post(
    "/stream",
    summary="Stream inference results",
    description="Streams NDJSON chunks representing sentences, evidence, scores, and completion. Content-Type is application/x-ndjson.",
)
def stream_inference(
    payload: InferenceRequest,
    response: Response,
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
):
    """Stream the inference process as NDJSON chunks.

    This endpoint applies the same pipeline stages as /inference/run but emits
    incremental results:
      - sentence: when a sentence is identified (with claim flag)
      - evidence: when evidence is gathered for a claim
      - score: final scored/labelled claim result
      - done: completion marker with a brief summary

    Request body:
      InferenceRequest JSON.

    Headers:
      - Authorization: Bearer <token> (or allowed anonymous based on env)
      - X-Client-Label: optional label for client/application
      - X-Request-ID: optional request id (server generates if missing)

    Returns:
      StreamingResponse with media type application/x-ndjson, emitting one JSON
      object per line for each event. May include empty-object heartbeats to keep
      the connection alive during long searches.
    """
    started = time.perf_counter()
    request_id = label.get("request_id")
    client_label = label.get("client_label")

    top_k = int(getattr(payload, "top_k", 5) or 5)
    search_limit = max(1, min(50, top_k))

    log_info(
        "inference.stream:start",
        event="inference_stream_start",
        request_id=request_id,
        client_label=client_label,
        top_k=top_k,
        search_limit=search_limit,
        language=payload.language,
    )

    with time_block("sentencize", request_id=request_id, client_label=client_label):
        raw_sentences: List[str] = get_sentences(payload.text or "")
    with time_block("claim_detection", request_id=request_id, client_label=client_label):
        claim_flags: List[bool] = detect_claims(raw_sentences)

    # Internal generator to yield events
    def _generate() -> Iterator[Dict[str, Any]]:
        seq = 0
        total = max(1, len(raw_sentences))
        # Emit sentence events
        for idx, (s, is_claim) in enumerate(zip(raw_sentences, claim_flags)):
            chunk = SentenceChunk(text=s, start_char=None, end_char=None, is_claim=bool(is_claim))
            progress = (idx + 1) / total
            sentence_event = chunk.model_dump()
            # Attach labels in event metadata
            sentence_event["_meta"] = {"request_id": request_id, "client_label": client_label}
            yield make_sentence_event(sentence=sentence_event, seq=seq, progress=progress)
            seq += 1

            if not is_claim:
                continue

            claim_preview = s[:120]
            # Evidence search per-claim
            with time_block(
                "claim_search",
                request_id=request_id,
                client_label=client_label,
                sentence_index=idx,
                claim_preview=claim_preview,
            ):
                evidence = search_for_claim(s, top_k=search_limit)

            ev_payload = {
                "claim": s,
                "sentence_index": idx,
                "evidence": evidence,
                "_meta": {"request_id": request_id, "client_label": client_label},
            }
            yield make_evidence_event(evidence=ev_payload, seq=seq, progress=progress)
            seq += 1

            # Scoring and labeling
            with time_block(
                "claim_score",
                request_id=request_id,
                client_label=client_label,
                sentence_index=idx,
                claim_preview=claim_preview,
            ):
                scored = categorize_and_score(claim=s, evidence=evidence, top_k=top_k)

            score_payload = {
                "claim": s,
                "sentence_index": idx,
                "supporting_evidence": scored.get("supporting_evidence", []),
                "refuting_evidence": scored.get("refuting_evidence", []),
                "score": float(scored.get("score", 0.0)),
                "label": scored.get("label"),
                "_meta": {"request_id": request_id, "client_label": client_label},
            }
            yield make_score_event(score=score_payload, seq=seq, progress=progress)
            seq += 1

        elapsed_ms = int((time.perf_counter() - started) * 1000.0)
        summary = {
            "status": "completed",
            "meta": {
                "processing_time_ms": elapsed_ms,
                "user": user,
                "label": label,
                "request_metadata": payload.metadata or {},
                "options": {
                    "top_k": top_k,
                    "search_limit": search_limit,
                    "language": payload.language,
                    "streaming": True,
                },
            },
            "_meta": {"request_id": request_id, "client_label": client_label},
        }
        yield make_done_event(summary=summary, seq=seq)

        log_info(
            "inference.stream:end",
            event="inference_stream_end",
            request_id=request_id,
            client_label=client_label,
            processing_time_ms=elapsed_ms,
            sentences=len(raw_sentences),
            claims=sum(1 for f in claim_flags if f),
        )

    # Use heartbeat streaming for long operations
    return as_fastapi_streaming_response(
        items=_generate(),
        media_type="application/x-ndjson",
        heartbeat=True,
        heartbeat_interval_sec=10.0,
    )
