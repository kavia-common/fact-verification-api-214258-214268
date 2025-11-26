from fastapi import APIRouter, Depends, Response
from typing import Any, Dict, List
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
@router.get(
    "/stream",
    summary="Stream inference results",
    description="Placeholder streaming endpoint to be implemented with chunked responses.",
)
def stream_inference(
    response: Response,
    user: Dict[str, Any] = Depends(auth_required),
    label: Dict[str, Any] = Depends(label_context),
) -> Dict[str, Any]:
    """Streaming placeholder. Will be replaced by Server-Sent Events or WebSocket.

    Parameters:
      - response: Response object so dependency header propagation is consistent.
      - user: injected by auth dependency.
      - label: injected by label dependency.

    Returns:
      A simple JSON response indicating streaming will be implemented later.
    """
    return {
        "status": "accepted",
        "message": "Streaming scaffolding in place. Implementation in a later step.",
        "user": user,
        "label": label,
    }
