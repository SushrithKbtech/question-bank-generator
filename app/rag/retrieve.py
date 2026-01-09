from __future__ import annotations
import re
from typing import List

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def _keyword_score(text: str, keywords: List[str]) -> int:
    t = _norm(text)
    return sum(1 for k in keywords if k in t)

def _build_keywords(query: str) -> List[str]:
    q = _norm(query)
    # domain-aware anchors (add more if needed)
    if "grover" in q:
        return ["grover", "search", "oracle", "amplitude", "iterations"]
    if "shor" in q:
        return ["shor", "factoring", "period", "modular", "fourier"]
    if "deutsch" in q:
        return ["deutsch", "jozsa", "oracle", "balanced", "constant"]
    if "bloch" in q:
        return ["bloch", "sphere", "qubit", "state vector", "angles"]
    # fallback: main words
    words = [w for w in re.findall(r"[a-zA-Z]{4,}", q)]
    return list(dict.fromkeys(words))[:8]

def retrieve_top_k_strict(
    collection,
    query: str,
    k: int = 5,
    allowed_source_types: list[str] | None = None,
):
    results = collection.similarity_search_with_score(query, k=max(25, k * 6))
    docs = [r[0].page_content for r in results]
    metas = [r[0].metadata for r in results]
    dists = [float(r[1]) for r in results]

    keywords = _build_keywords(query)

    candidates = []
    for doc, meta, dist in zip(docs, metas, dists):
        source_type = (meta or {}).get("source_type", "material")
        if allowed_source_types and source_type not in allowed_source_types:
            continue
        score = _keyword_score(doc, keywords)
        candidates.append({
            "text": doc,
            "source": (meta or {}).get("source"),
            "page": (meta or {}).get("page"),
            "source_type": source_type,
            "distance": float(dist),
            "kw_score": score,
        })

    # Gate: require at least 1 keyword hit unless query is extremely generic
    if len(keywords) >= 3:
        candidates = [c for c in candidates if c["kw_score"] >= 1]

    candidates.sort(key=lambda x: (-x["kw_score"], x["distance"]))
    return candidates[:k]
