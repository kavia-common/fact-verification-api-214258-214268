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
    user: Dict[str, Any] = Depends(auth_required),
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

    # Respect options: derive limits with safe bounds.
    top_k = int(getattr(payload, "top_k", 5) or 5)
    # search_limit (number of results per claim) is mapped to top_k from request.
    search_limit = max(1, min(50, top_k))

    # 1) Sentence splitting
    raw_sentences: List[str] = get_sentences(payload.text or "")
    # 2) Claim detection
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
        # Search
        evidence = search_for_claim(s, top_k=search_limit)
        # Score & categorize
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
    top_k = int(getattr(payload, "top_k", 5) or 5)
    search_limit = max(1, min(50, top_k))

    raw_sentences: List[str] = get_sentences(payload.text or "")
    claim_flags: List[bool] = detect_claims(raw_sentences)

    # Internal generator to yield events
    def _generate() -> Iterator[Dict[str, Any]]:
        seq = 0
        total = max(1, len(raw_sentences))
        # Emit sentence events
        for idx, (s, is_claim) in enumerate(zip(raw_sentences, claim_flags)):
            chunk = SentenceChunk(text=s, start_char=None, end_char=None, is_claim=bool(is_claim))
            progress = (idx + 1) / total
            yield make_sentence_event(sentence=chunk.model_dump(), seq=seq, progress=progress)
            seq += 1

            if not is_claim:
                continue

            # Evidence search per-claim
            evidence = search_for_claim(s, top_k=search_limit)
            ev_payload = {
                "claim": s,
                "sentence_index": idx,
                "evidence": evidence,
            }
            yield make_evidence_event(evidence=ev_payload, seq=seq, progress=progress)
            seq += 1

            # Scoring and labeling
            scored = categorize_and_score(claim=s, evidence=evidence, top_k=top_k)
            score_payload = {
                "claim": s,
                "sentence_index": idx,
                "supporting_evidence": scored.get("supporting_evidence", []),
                "refuting_evidence": scored.get("refuting_evidence", []),
                "score": float(scored.get("score", 0.0)),
                "label": scored.get("label"),
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
        }
        yield make_done_event(summary=summary, seq=seq)

    # Use heartbeat streaming for long operations
    return as_fastapi_streaming_response(
        items=_generate(),
        media_type="application/x-ndjson",
        heartbeat=True,
        heartbeat_interval_sec=10.0,
    )
