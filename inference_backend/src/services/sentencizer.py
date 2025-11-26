from typing import List


# PUBLIC_INTERFACE
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences (placeholder implementation).

    This is a scaffold; in a subsequent step we will integrate spaCy's sentencizer.
    """
    if not text:
        return []
    # Naive split for scaffolding; replace with spaCy in a later step.
    return [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
