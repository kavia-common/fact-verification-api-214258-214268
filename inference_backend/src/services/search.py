from typing import List, Dict, Any


# PUBLIC_INTERFACE
def web_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform a web search (placeholder).

    In a future step, integrate with a real search API or local index.
    """
    if not query:
        return []
    # Placeholder search results
    return [
        {
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/{i+1}",
            "snippet": "Example snippet content.",
            "score": 1.0 / (i + 1),
        }
        for i in range(top_k)
    ]
