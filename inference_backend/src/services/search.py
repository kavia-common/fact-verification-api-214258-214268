import os
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterable

import httpx


def _clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    return s if s else None


def _norm_result(url: Optional[str], title: Optional[str], snippet: Optional[str], score: float = 0.0,
                 source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Normalize a single result to the EvidenceItem-like dict."""
    url = _clean_text(url)
    title = _clean_text(title)
    if not url or not title:
        return None
    item: Dict[str, Any] = {
        "title": title,
        "url": url,
        "snippet": _clean_text(snippet),
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
    }
    if source:
        item["source"] = source
    if metadata:
        item["metadata"] = metadata
    return item


# Provider Interface
class SearchProvider(ABC):
    """Abstract search provider interface."""

    # PUBLIC_INTERFACE
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Execute a search and return normalized results (url, title, snippet, score, source, metadata)."""
        raise NotImplementedError


class BingWebSearchProvider(SearchProvider):
    """Bing Web Search API provider using an API key.

    Environment:
      - BING_API_KEY or SEARCH_API_KEY: API key for Bing Web Search
      - Optional: BING_ENDPOINT (defaults to official endpoint)
    Docs: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
    """

    def __init__(self, api_key: str, endpoint: Optional[str] = None, market: str = "en-US", safe: str = "Off"):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.bing.microsoft.com/v7.0/search"
        self.market = market
        self.safe = safe

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "mkt": self.market,
            "count": max(1, min(50, int(top_k))),
            "safeSearch": self.safe,
            # Keep response slim
            "textDecorations": False,
            "textFormat": "Raw",
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self.endpoint, headers=headers, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return []
        web_pages = (data or {}).get("webPages", {})
        values: Iterable[Dict[str, Any]] = web_pages.get("value", []) or []
        results: List[Dict[str, Any]] = []
        for i, v in enumerate(values):
            title = v.get("name") or v.get("title")
            url = v.get("url")
            snippet = v.get("snippet") or v.get("about") or v.get("description")
            score = 1.0 / (i + 1)
            meta = {
                "rank": i + 1,
                "language": web_pages.get("language"),
                "total_estimated_matches": web_pages.get("totalEstimatedMatches"),
            }
            item = _norm_result(url, title, snippet, score=score, source="bing", metadata=meta)
            if item:
                results.append(item)
            if len(results) >= top_k:
                break
        return results


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo keyless provider via html/jsonlite endpoints (best-effort)."""

    def __init__(self):
        # Use html endpoint with q param; also add site: filters for Wikipedia in query formation if needed
        self.endpoint_html = "https://duckduckgo.com/html/"

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []
        params = {"q": query}
        results: List[Dict[str, Any]] = []
        try:
            # jsonlite API is deprecated; use html endpoint and scrape minimal anchors via naive approach
            with httpx.Client(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
                resp = client.get(self.endpoint_html, params=params)
                resp.raise_for_status()
                html = resp.text
        except Exception:
            return []

        # Naive extraction: look for result blocks <a rel="nofollow" class="result__a" href="...">Title</a>
        # Avoid full HTML parser to keep deps minimal.
        # This is best-effort and may change with DDG markup; graceful degradation is acceptable.
        anchors: List[Dict[str, str]] = []
        try:
            # Very lightweight pattern search
            # Find occurrences of 'result__a' links
            marker = 'class="result__a"'
            pos = 0
            while True:
                idx = html.find(marker, pos)
                if idx == -1:
                    break
                # find href="
                href_idx = html.rfind('href="', 0, idx)
                if href_idx == -1:
                    pos = idx + len(marker)
                    continue
                href_start = href_idx + len('href="')
                href_end = html.find('"', href_start)
                if href_end == -1:
                    pos = idx + len(marker)
                    continue
                href = html[href_start:href_end]

                # title between > and </a>
                title_start = html.find(">", idx) + 1
                title_end = html.find("</a>", title_start)
                if title_end == -1:
                    pos = idx + len(marker)
                    continue
                raw_title = html[title_start:title_end]
                # Strip tags inside title (very lightweight)
                title = (
                    raw_title.replace("<b>", "")
                    .replace("</b>", "")
                    .replace("&amp;", "&")
                    .replace("&#39;", "'")
                    .replace("&quot;", '"')
                )
                anchors.append({"href": href, "title": title})
                pos = title_end + 4
        except Exception:
            anchors = []

        for i, a in enumerate(anchors):
            # DDG wraps links with '/l/?kh=-1&uddg=<encoded_url>'
            href = a.get("href") or ""
            url = href
            if "uddg=" in href:
                # Extract uddg param
                try:
                    parsed = urllib.parse.urlparse(href)
                    q = urllib.parse.parse_qs(parsed.query)
                    uddg = q.get("uddg", [None])[0]
                    if uddg:
                        url = urllib.parse.unquote(uddg)
                except Exception:
                    pass
            title = a.get("title")
            item = _norm_result(url, title, snippet=None, score=1.0 / (i + 1), source="duckduckgo")
            if item:
                results.append(item)
            if len(results) >= top_k:
                break
        return results


class WikipediaSearchProvider(SearchProvider):
    """Wikipedia keyless search provider using opensearch and query APIs."""

    def __init__(self, language: str = "en"):
        self.api = f"https://{language}.wikipedia.org/w/api.php"

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []
        # First use opensearch to get titles and descriptions
        params = {
            "action": "opensearch",
            "search": query,
            "limit": max(1, min(50, int(top_k))),
            "namespace": 0,
            "format": "json",
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self.api, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return []
        results: List[Dict[str, Any]] = []
        try:
            titles = data[1] if len(data) > 1 else []
            descriptions = data[2] if len(data) > 2 else []
            urls = data[3] if len(data) > 3 else []
            for i, (t, d, u) in enumerate(zip(titles, descriptions, urls)):
                item = _norm_result(u, t, d, score=1.0 / (i + 1), source="wikipedia")
                if item:
                    results.append(item)
                if len(results) >= top_k:
                    break
        except Exception:
            return []
        return results


def _formulate_web_query_from_claim(claim: str) -> str:
    """Create a concise search query from a claim sentence."""
    text = (claim or "").strip()
    if not text:
        return ""
    # Light normalization: remove surrounding quotes and collapse whitespace
    text = text.strip(' "\'\n\t')
    text = " ".join(text.split())
    # Heuristic tweaks:
    # - If sentence ends with period, drop it for better search
    if text.endswith("."):
        text = text[:-1]
    # Prefer shorter queries by limiting to first ~20 words
    words = text.split()
    if len(words) > 20:
        text = " ".join(words[:20])
    return text


def _get_provider() -> SearchProvider:
    """Select a provider based on SEARCH_PROVIDER and available environment variables.

    Environment variables:
      - SEARCH_PROVIDER: "auto" | "bing" | "duckduckgo" | "wikipedia" (case-insensitive). Default "auto".
      - BING_API_KEY / SEARCH_API_KEY: API key for Bing Web Search (required if using "bing").
      - BING_ENDPOINT: Optional custom endpoint for Bing.

    SEARCH_PROVIDER values (case-insensitive):
      - "bing": Use BingWebSearchProvider (requires BING_API_KEY or SEARCH_API_KEY).
      - "duckduckgo": Use DuckDuckGoProvider (keyless).
      - "wikipedia": Use WikipediaSearchProvider (keyless).
      - "auto" or unset: Prefer Bing if key provided; otherwise fall back to DuckDuckGo+Wikipedia composite.

    Note: See inference_backend/.env.example for configuration examples.
    """
    # Explicit override
    choice = (os.getenv("SEARCH_PROVIDER") or "auto").strip().lower()

    if choice == "bing":
        api_key = os.getenv("BING_API_KEY") or os.getenv("SEARCH_API_KEY")
        if api_key:
            endpoint = os.getenv("BING_ENDPOINT")
            return BingWebSearchProvider(api_key=api_key, endpoint=endpoint)
        # If requested bing but missing key, gracefully fall back to composite
        return CompositeProvider([DuckDuckGoProvider(), WikipediaSearchProvider()])

    if choice == "duckduckgo":
        return DuckDuckGoProvider()

    if choice == "wikipedia":
        return WikipediaSearchProvider()

    # auto/default
    api_key = os.getenv("BING_API_KEY") or os.getenv("SEARCH_API_KEY")
    if api_key:
        endpoint = os.getenv("BING_ENDPOINT")
        return BingWebSearchProvider(api_key=api_key, endpoint=endpoint)
    return CompositeProvider([DuckDuckGoProvider(), WikipediaSearchProvider()])


class CompositeProvider(SearchProvider):
    """Try multiple providers until enough results are gathered."""

    def __init__(self, providers: List[SearchProvider]):
        self.providers = providers

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        aggregated: List[Dict[str, Any]] = []
        seen_urls = set()
        for provider in self.providers:
            try:
                results = provider.search(query, top_k=top_k)
            except Exception:
                results = []
            for r in results:
                url = r.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                aggregated.append(r)
                if len(aggregated) >= top_k:
                    return aggregated
        return aggregated


# PUBLIC_INTERFACE
def web_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Perform a web search using a pluggable provider system.

    Provider selection:
      - If SEARCH_PROVIDER is set, choose the specified provider (bing/duckduckgo/wikipedia/auto).
      - If BING_API_KEY or SEARCH_API_KEY is present (or SEARCH_PROVIDER=bing), uses Bing Web Search API.
      - Otherwise, falls back to keyless providers (DuckDuckGo HTML and Wikipedia opensearch).

    Returns normalized list of dicts:
      - title: str
      - url: str
      - snippet: Optional[str]
      - score: float
      - source: Optional[str]
      - metadata: Optional[dict]
    """
    if not query:
        return []
    provider = _get_provider()
    try:
        return provider.search(query=query, top_k=top_k)
    except Exception:
        return []


# PUBLIC_INTERFACE
def search_for_claim(claim: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search evidence for a claim sentence.

    This helper formulates a query from a claim and delegates to web_search.

    Parameters:
      - claim: a likely factual sentence
      - top_k: desired number of results

    Returns:
      List of normalized results suitable for EvidenceItem.
    """
    q = _formulate_web_query_from_claim(claim)
    if not q:
        return []
    return web_search(q, top_k=top_k)
