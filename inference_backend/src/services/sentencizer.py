from functools import lru_cache
import re
from typing import List

try:
    import spacy
    from spacy.language import Language
except Exception:  # pragma: no cover - import fallback path
    spacy = None
    Language = None  # type: ignore[assignment]


_SENTENCE_REGEX = re.compile(
    r"""
    # Split on ., !, or ? followed by space/newline and a capital or quote, preserving abbreviations
    (?<!\b[A-Z])           # crude guard against single-letter initial abbreviations like "A."
    (?<=[\.\!\?])          # end punctuation
    \s+                    # whitespace after punctuation
    (?=(["'(\[]?[A-Z0-9])) # next sentence likely starts with capital/number/quote/bracket
    """,
    re.VERBOSE,
)


@lru_cache(maxsize=1)
def _get_spacy_pipeline():
    """Create and cache a lightweight spaCy pipeline with sentencizer.

    Uses spacy.blank('en') with the rule-based sentencizer to avoid model downloads.
    If spaCy is unavailable, returns None so the caller can use regex fallback.
    """
    if spacy is None:
        return None
    try:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        # If anything goes wrong (e.g., environment), fallback will be used.
        return None


def _regex_split(text: str) -> List[str]:
    """Regex-based sentence splitting fallback with basic heuristics."""
    if not text:
        return []
    # Normalize whitespace
    clean = re.sub(r"\s+", " ", text.strip())
    # A simpler approach: split on (?<=[.!?]) + space/newline and then strip.
    simple = re.split(r"(?<=[\.\!\?])\s+", clean)
    sentences = [s.strip() for s in simple if s and not s.isspace()]
    return sentences


# PUBLIC_INTERFACE
def get_sentences(text: str) -> List[str]:
    """Return a list of sentence strings for the given text.

    The function attempts to use a cached spaCy sentencizer (spacy.blank('en') + sentencizer).
    If spaCy is unavailable or fails to initialize, it falls back to a robust regex-based splitter.

    Parameters:
      text: Raw input text.

    Returns:
      List of sentence strings in reading order.
    """
    if not text:
        return []

    nlp = _get_spacy_pipeline()
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences = [s.text.strip() for s in doc.sents if s.text and not s.text.isspace()]
            if sentences:
                return sentences
        except Exception:
            # Fall back to regex if spaCy processing encounters errors
            pass

    return _regex_split(text)


# PUBLIC_INTERFACE
def split_into_sentences(text: str) -> List[str]:
    """Deprecated shim: use get_sentences instead.

    Kept for backward compatibility with earlier scaffolding.
    """
    return get_sentences(text)
