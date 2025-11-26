from __future__ import annotations

import math
import re
from typing import List, Dict, Any, Tuple, Optional, Iterable

# Simple global stopwords set to improve token matching without extra deps.
_STOPWORDS = {
    "a", "an", "and", "or", "the", "to", "of", "in", "on", "for", "with", "by", "is", "are", "was", "were",
    "be", "been", "being", "as", "at", "from", "that", "this", "it", "its", "their", "his", "her", "they",
    "them", "we", "you", "your", "i", "me", "my", "our", "ours", "but", "if", "then", "so", "than", "too",
    "very", "can", "could", "should", "would", "will", "may", "might", "not", "no"
}
_TOKENIZER = re.compile(r"[A-Za-z0-9]+")

_NEGATION_CUES = {
    "not", "no", "never", "neither", "nor", "without", "none", "cannot", "can't", "isn't", "aren't",
    "wasn't", "weren't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "can't",
}

_CONTRADICTION_CUES = {
    "contradict", "refute", "refutes", "refuted", "false", "falsely", "debunk", "debunks", "debunked",
    "myth", "hoax", "incorrect", "misleading", "disprove", "disproves", "disproved",
}


def _norm_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _tokens(text: str) -> List[str]:
    # Lowercase, alnum tokens
    toks = [t.lower() for t in _TOKENIZER.findall(text)]
    # filter stopwords but keep numbers and longer keywords
    return [t for t in toks if (t not in _STOPWORDS and (len(t) > 2 or t.isdigit()))]


def _build_doc(text: str, title: Optional[str]) -> List[str]:
    parts = []
    if title:
        parts.append(title)
    if text:
        parts.append(text)
    return _tokens(" ".join(parts))


def _idf(term: str, df_map: Dict[str, int], N: int) -> float:
    # BM25-like idf with smoothing
    df = max(1, df_map.get(term, 0))
    return math.log((N - df + 0.5) / (df + 0.5) + 1.0)


def _bm25_lite(query_tokens: List[str], doc_tokens: List[str], avgdl: float, df_map: Dict[str, int], N: int,
               k1: float = 1.2, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    # Term frequencies in doc
    tf: Dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    dl = len(doc_tokens)
    score = 0.0
    for q in query_tokens:
        if q not in tf:
            continue
        idf = _idf(q, df_map, N)
        denom = tf[q] + k1 * (1 - b + b * (dl / (avgdl if avgdl > 0 else 1.0)))
        score += idf * ((tf[q] * (k1 + 1)) / denom)
    # normalize lightly
    return max(0.0, score)


def _jaccard(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    qs, ds = set(query_tokens), set(doc_tokens)
    inter = len(qs & ds)
    union = len(qs | ds)
    if union == 0:
        return 0.0
    return inter / union


def _corpus_stats(docs: List[List[str]]) -> Tuple[float, Dict[str, int], int]:
    # average doc length, document frequency map, total docs
    if not docs:
        return 0.0, {}, 0
    N = len(docs)
    total_len = 0
    df_map: Dict[str, int] = {}
    for d in docs:
        total_len += len(d)
        for t in set(d):
            df_map[t] = df_map.get(t, 0) + 1
    avgdl = total_len / max(1, N)
    return avgdl, df_map, N


def _stance_heuristic(claim: str, title: Optional[str], snippet: Optional[str]) -> float:
    """
    Simple stance heuristic:
      - If we detect explicit contradiction/negation cues in title/snippet relative to claim, lean negative.
      - If claim has negation and evidence lacks it (or vice-versa), lean opposite a bit.
    Returns a stance multiplier in [-1, 1], where positive means supporting, negative means refuting.
    """
    c = _norm_text(claim).lower()
    t = _norm_text(title).lower()
    s = _norm_text(snippet).lower()

    # Presence of explicit contradiction cues strongly suggests refuting
    has_contradict = any(w in t or w in s for w in _CONTRADICTION_CUES)
    if has_contradict:
        return -0.9

    # Count negation cues
    claim_neg = sum(1 for w in _NEGATION_CUES if w in c) > 0
    ev_neg = sum(1 for w in _NEGATION_CUES if (w in t or w in s)) > 0

    # If both have or both lack negation, treat as supporting; if mismatch, refuting lightly.
    if claim_neg == ev_neg:
        return 0.6  # moderately supportive
    else:
        return -0.6  # moderately refuting


def _similarity_score(claim: str, title: Optional[str], snippet: Optional[str],
                      corpus_avgdl: float, corpus_df: Dict[str, int], corpus_N: int) -> float:
    """
    Combine BM25-lite and Jaccard for a stable lightweight similarity.
    """
    q = _tokens(_norm_text(claim))
    d = _build_doc(_norm_text(snippet), _norm_text(title))
    if not q or not d:
        return 0.0
    bm25 = _bm25_lite(q, d, corpus_avgdl, corpus_df, corpus_N)
    jac = _jaccard(q, d)
    # Weighted combination; BM25 dominates, Jaccard gives stability for tiny docs
    return 0.8 * bm25 + 0.2 * jac


def _score_items_for_claim(claim: str, items: List[Dict[str, Any]],
                           corpus_avgdl: float, corpus_df: Dict[str, int], corpus_N: int) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for it in items:
        title = it.get("title") or ""
        snippet = it.get("snippet") or ""
        sim = _similarity_score(claim, title, snippet, corpus_avgdl, corpus_df, corpus_N)
        stance = _stance_heuristic(claim, title, snippet)
        # Final per-item score is similarity scaled by stance direction
        # Keep also raw similarity for ranking support/refute separately.
        enriched = dict(it)
        enriched["similarity"] = float(sim)
        enriched["stance"] = float(stance)
        enriched["score"] = float(sim * abs(stance))  # relevance magnitude
        enriched["support_score"] = float(sim * max(0.0, stance))
        enriched["refute_score"] = float(sim * max(0.0, -stance))
        scored.append(enriched)
    return scored


def _gather_corpus_docs(items: Iterable[Dict[str, Any]]) -> List[List[str]]:
    docs: List[List[str]] = []
    for it in items:
        title = _norm_text(it.get("title") or "")
        snippet = _norm_text(it.get("snippet") or "")
        docs.append(_build_doc(snippet, title))
    return docs


# PUBLIC_INTERFACE
def rank_and_label_evidence_for_claim(
    claim: str,
    evidence: List[Dict[str, Any]],
    top_k_support: int = 3,
    top_k_refute: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, str]:
    """Rank evidence for a claim and determine a simple label.

    PUBLIC_INTERFACE
    Parameters:
      - claim: Target sentence/claim text.
      - evidence: List of normalized evidence dicts (title, url, snippet, ...).
      - top_k_support: Number of top supporting items to return.
      - top_k_refute: Number of top refuting items to return.

    Returns:
      (supporting_evidence, refuting_evidence, aggregate_score, label)
      where label in {"SUPPORTED", "REFUTED", "NEI"}.

    Notes:
      - Uses token-based BM25-lite + Jaccard similarity and a negation/contradiction-based stance heuristic.
      - Aggregate score is sum(top support scores) - sum(top refute scores).
    """
    ev = evidence or []
    if not claim or not ev:
        return [], [], 0.0, "NEI"

    corpus_docs = _gather_corpus_docs(ev)
    avgdl, df_map, N = _corpus_stats(corpus_docs)

    scored = _score_items_for_claim(claim, ev, avgdl, df_map, N)

    # Partition by stance sign
    support_items = [x for x in scored if x.get("support_score", 0.0) > 0]
    refute_items = [x for x in scored if x.get("refute_score", 0.0) > 0]

    # Sort each group by their respective directional score
    support_items.sort(key=lambda x: (x.get("support_score", 0.0), x.get("similarity", 0.0)), reverse=True)
    refute_items.sort(key=lambda x: (x.get("refute_score", 0.0), x.get("similarity", 0.0)), reverse=True)

    top_support = support_items[: max(0, int(top_k_support))]
    top_refute = refute_items[: max(0, int(top_k_refute))]

    support_sum = sum(x.get("support_score", 0.0) for x in top_support)
    refute_sum = sum(x.get("refute_score", 0.0) for x in top_refute)
    aggregate = float(support_sum - refute_sum)

    # Label decision heuristic:
    # - If aggregate strongly > 0 and at least one supporting item, SUPPORTED
    # - If aggregate strongly < 0 and at least one refuting item, REFUTED
    # - Otherwise NEI
    # Thresholds are loose to avoid over-claiming with noisy search results.
    label: str
    if support_sum >= max(0.6, 1.2 * refute_sum) and len(top_support) > 0:
        label = "SUPPORTED"
    elif refute_sum >= max(0.6, 1.2 * support_sum) and len(top_refute) > 0:
        label = "REFUTED"
    else:
        label = "NEI"

    # Clean items: ensure required fields for downstream schemas exist
    def _minimalize(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in items:
            out.append({
                "title": it.get("title") or "",
                "url": it.get("url") or "",
                "snippet": it.get("snippet"),
                "score": float(it.get("score", 0.0)),
                "source": it.get("source"),
                "metadata": it.get("metadata"),
            })
        return out

    return _minimalize(top_support), _minimalize(top_refute), aggregate, label


# PUBLIC_INTERFACE
def categorize_and_score(
    claim: str,
    evidence: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    """Convenience wrapper returning a dict ready to populate ClaimResult fields.

    PUBLIC_INTERFACE
    Parameters:
      - claim: sentence/claim text
      - evidence: normalized evidence items
      - top_k: number of items to keep for both support and refute lists

    Returns:
      Dict with keys: supporting_evidence, refuting_evidence, score, label
    """
    sup, ref, agg, label = rank_and_label_evidence_for_claim(
        claim=claim, evidence=evidence, top_k_support=top_k, top_k_refute=top_k
    )
    return {
        "supporting_evidence": sup,
        "refuting_evidence": ref,
        "score": float(agg),
        "label": label,
    }


# PUBLIC_INTERFACE
def score_evidence(supporting: List[Dict[str, Any]], refuting: List[Dict[str, Any]]) -> float:
    """Aggregate claim score from already separated supporting/refuting evidence.

    Computes: sum(supporting scores) - sum(refuting scores), where `score` is assumed
    to be the relevance magnitude of each item.
    """
    support_sum = sum(float(item.get("score", 0.0)) for item in (supporting or []))
    refute_sum = sum(float(item.get("score", 0.0)) for item in (refuting or []))
    return float(support_sum - refute_sum)
