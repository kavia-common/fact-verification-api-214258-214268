from typing import List


# PUBLIC_INTERFACE
def is_claim(sentence: str) -> bool:
    """Determine if a sentence is a claim (placeholder).

    Will be replaced by a more robust model-based or heuristic approach later.
    """
    if not sentence:
        return False
    # Simple heuristic placeholder: contains a verb-like pattern.
    tokens = sentence.lower().split()
    return any(t in tokens for t in ["is", "are", "was", "were", "has", "have", "claims", "states"])


# PUBLIC_INTERFACE
def filter_claims(sentences: List[str]) -> List[str]:
    """Filter sentences to those that are likely claims (placeholder)."""
    return [s for s in sentences if is_claim(s)]
