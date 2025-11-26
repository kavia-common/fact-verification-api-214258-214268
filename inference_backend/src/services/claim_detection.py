from typing import List

# Minimal lists of cue words to keep heuristics lightweight and dependency-free.
_ASSERTIVE_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had",
    "does", "do", "did",
    "claims", "claim", "states", "state", "reports", "report",
    "shows", "show", "demonstrates", "demonstrate", "confirms", "confirm",
    "suggests", "suggest",
}
_MODAL_VERBS = {"will", "would", "can", "could", "should", "must", "may", "might"}
_NEGATION = {"not", "no", "never", "none", "cannot", "can't", "won't", "isn't", "aren't", "doesn't", "don't", "didn't"}
_NON_CLAIM_LEADS = {
    # Interrogatives and common question openers
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    # Imperatives / instructions often not factual claims
    "please", "note", "remember", "consider", "imagine", "let's", "let", "ensure",
}
_REPORTING_VERBS = {"said", "says", "according", "reported", "reports"}  # can still be claims, but useful cues
_NUMERIC_TOKENS = set("0123456789")

def _has_terminal_punctuation(text: str) -> bool:
    return any(text.rstrip().endswith(p) for p in [".", "!", "?"])

def _looks_like_question(text: str, tokens: List[str]) -> bool:
    # Ends with question mark or starts with interrogatives
    if text.strip().endswith("?"):
        return True
    if tokens and tokens[0] in _NON_CLAIM_LEADS:
        return True
    # Heuristic: presence of auxiliary inversion patterns like "is/are/can ... ?" already covered by '?'
    return False

def _has_verb_like(tokens: List[str]) -> bool:
    # Very lightweight proxy for verb presence using assertive/modal lexicons
    return any(t in _ASSERTIVE_VERBS or t in _MODAL_VERBS for t in tokens)

def _has_numeric_or_named_like(tokens: List[str], text: str) -> bool:
    # Numeric cue: digits in sentence
    if any(ch in _NUMERIC_TOKENS for ch in text):
        return True
    # Crude NER cue: TitleCase tokens inside the sentence (not at start only)
    # We consider words with an internal uppercase letter (e.g., "New", "York", "NASA")
    # but avoid counting the first token only (which may be capitalized by sentence case)
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if tok[:1].isupper():
            return True
        # ALLCAPS (short acronyms) can also be a cue
        if len(tok) > 1 and tok.isupper():
            return True
    return False

def _strip_quotes(text: str) -> str:
    # Remove surrounding quotes/brackets to avoid misclassification
    stripped = text.strip().strip("\"'“”‘’()[]{}")
    return stripped if stripped else text.strip()


# PUBLIC_INTERFACE
def is_claim(sentence: str) -> bool:
    """Determine if a sentence is likely a factual claim using lightweight heuristics.

    Heuristics:
      - Non-question (avoid sentences ending with '?' or starting with interrogatives).
      - Presence of verb-like cues (assertive or modal verbs).
      - Numeric or name-entity-like cues (digits, TitleCase or ALLCAPS tokens after the first word).
      - Negative or reporting constructs do not invalidate claims, but contribute indirectly.
      - Very short or very long sentences are de-prioritized (length guard).

    Design:
      This function is intentionally dependency-free and fast. It should be considered
      a pluggable component so it can be replaced by an ML classifier in the future
      without requiring API changes.

    Parameters:
      sentence: Input sentence text.

    Returns:
      True if heuristically considered a claim; False otherwise.
    """
    if not sentence:
        return False

    text = _strip_quotes(sentence)
    # Quick length guardrails
    n_chars = len(text)
    if n_chars < 6:  # too short to be a claim
        return False
    if n_chars > 2000:  # extremely long single "sentence" likely not a clean sentence
        return False

    tokens_raw = text.split()
    tokens = [t.strip(".,;:!?\"'()[]{}").lower() for t in tokens_raw if t.strip()]
    if not tokens:
        return False

    # Avoid questions
    if _looks_like_question(text, tokens):
        return False

    # Require some verb-like presence
    verbish = _has_verb_like(tokens)

    # Numeric or name-like cues can strengthen claim likelihood
    has_numeric_or_named = _has_numeric_or_named_like(tokens_raw, text)

    # Terminal punctuation is a mild positive signal (esp. statements)
    terminal = _has_terminal_punctuation(text)

    # Combine signals with a simple scoring rule to keep behavior transparent.
    score = 0
    if verbish:
        score += 2
    if has_numeric_or_named:
        score += 1
    if terminal:
        score += 1

    # Soft penalties
    if tokens and tokens[0] in _REPORTING_VERBS:
        # "According to X" can still be claim-like; give a tiny penalty to require another cue
        score -= 1

    # Threshold chosen to balance precision/recall for heuristic use
    return score >= 2


# PUBLIC_INTERFACE
def filter_claims(sentences: List[str]) -> List[str]:
    """Filter sentences to those that are likely claims using is_claim() heuristics."""
    return [s for s in sentences if is_claim(s)]


# PUBLIC_INTERFACE
def detect_claims(sentences: List[str]) -> List[bool]:
    """Return a parallel list of booleans indicating whether each sentence is a claim.

    This keeps the interface pluggable for a future ML-based detector that can score
    batches of sentences efficiently.

    Parameters:
      sentences: List of sentence strings.

    Returns:
      List[bool]: True for sentences considered claims, False otherwise.
    """
    if not sentences:
        return []
    return [is_claim(s or "") for s in sentences]
