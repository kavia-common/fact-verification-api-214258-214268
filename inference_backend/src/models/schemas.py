"""
Pydantic schema definitions for the Fact Verification Inference API.

This module contains the request and response models used by the API routes
as well as streaming chunk representations for future server-sent events or
chunked transfer responses.
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# PUBLIC_INTERFACE
class InferenceRequest(BaseModel):
    """Request payload for running inference on an input text.

    Attributes:
      text: The raw input text that will be split into sentences and analyzed.
      language: Optional language code (e.g., 'en'); can guide sentencizer/tokenizer choices.
      top_k: Number of search results to fetch per claim for evidence gathering.
      streaming: If true, server may stream incremental results instead of a single response.
      metadata: Arbitrary client-provided metadata echoed back for traceability.
    """
    text: str = Field(..., description="Raw input text to analyze for claims and evidence.")
    language: Optional[str] = Field(
        default=None,
        description="Optional language code for processing (e.g., 'en').",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top search results to retrieve per claim."
    )
    streaming: bool = Field(
        default=False,
        description="If true, the server may stream incremental results."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional user metadata to be echoed in responses."
    )


# PUBLIC_INTERFACE
class SentenceChunk(BaseModel):
    """A sentence chunk produced by sentencizer.

    Attributes:
      text: Sentence text content.
      start_char: Start character offset in the original text.
      end_char: End character offset in the original text.
      is_claim: Heuristic/model determination whether this sentence is a claim.
    """
    text: str = Field(..., description="Sentence text.")
    start_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start character index of the sentence in the original text."
    )
    end_char: Optional[int] = Field(
        default=None,
        ge=0,
        description="End character index of the sentence in the original text."
    )
    is_claim: bool = Field(..., description="Whether the sentence is identified as a claim.")


# PUBLIC_INTERFACE
class EvidenceItem(BaseModel):
    """Evidence item from a search result or knowledge base.

    Attributes:
      title: Title of the evidence source.
      url: URL pointing to the evidence.
      snippet: Short text snippet from the source.
      score: Relevance or support/refute score for the evidence item.
      source: Optional source/provider label (e.g., 'web', 'kb').
      metadata: Optional auxiliary information returned by the provider.
    """
    title: str = Field(..., description="Title of the evidence source.")
    url: str = Field(..., description="URL of the evidence source.")
    snippet: Optional[str] = Field(default=None, description="Short snippet from the source.")
    score: float = Field(default=0.0, description="Relevance score for this evidence item.")
    source: Optional[str] = Field(default=None, description="Provider/source label (e.g., 'web').")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific metadata (e.g., ranking, domain, date)."
    )


# PUBLIC_INTERFACE
class ClaimResult(BaseModel):
    """Result for a single claim including supporting and refuting evidence.

    Attributes:
      claim: The claim text (often the sentence itself).
      sentence_index: Index of the sentence in the input (0-based).
      supporting_evidence: List of items that support the claim.
      refuting_evidence: List of items that refute the claim.
      score: Aggregate score computed from evidence (support - refute or model-specific).
      label: Optional label for the claim (e.g., 'SUPPORTED', 'REFUTED', 'NEI').
    """
    claim: str = Field(..., description="The claim text.")
    sentence_index: int = Field(..., ge=0, description="Index of the sentence in the input text.")
    supporting_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items that support the claim."
    )
    refuting_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items that refute the claim."
    )
    score: float = Field(
        default=0.0,
        description="Aggregate score for the claim based on evidence."
    )
    label: Optional[Literal["SUPPORTED", "REFUTED", "NEI"]] = Field(
        default=None,
        description="Optional claim label (e.g., SUPPORTED/REFUTED/NEI)."
    )


# PUBLIC_INTERFACE
class InferenceResponse(BaseModel):
    """Full response for an inference run.

    Attributes:
      status: Processing status of the request.
      sentences: All sentences detected in the input, with claim flags.
      claims: Results for sentences determined to be claims.
      metadata: Echoed metadata from request or processing details.
      message: Optional human-readable message about the run.
    """
    status: Literal["accepted", "completed", "error"] = Field(
        ...,
        description="Status of the operation."
    )
    sentences: List[SentenceChunk] = Field(
        default_factory=list,
        description="All sentences detected from the input text."
    )
    claims: List[ClaimResult] = Field(
        default_factory=list,
        description="Results for detected claims."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata associated to this run (request or processing details)."
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional message about the run or errors."
    )


# PUBLIC_INTERFACE
class StreamChunk(BaseModel):
    """A chunk of data for streaming inference results.

    Attributes:
      event: The type of event (e.g., 'sentence', 'claim', 'complete', 'error').
      data: The payload associated with the event:
            - sentence: SentenceChunk
            - claim: ClaimResult
            - complete: InferenceResponse or summary data
            - error: error message or structure
      seq: Monotonic sequence number for chunk ordering.
      progress: Optional progress indicator in [0, 1].
    """
    event: Literal["sentence", "claim", "complete", "error"] = Field(
        ...,
        description="Event type of the streamed chunk."
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Payload for the event; structure depends on the event type."
    )
    seq: int = Field(
        ...,
        ge=0,
        description="Sequence number for ordering of chunks."
    )
    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional normalized progress indicator."
    )
