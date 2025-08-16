# src/utils_citation.py
from typing import Dict, Any

def format_ieee(metadata: Dict[str, Any]) -> str:
    """
    Build a short IEEE-like citation string from metadata produced in indexing.
    metadata expected keys: title, link, published, authors (maybe list)
    """
    title = metadata.get("title") or metadata.get("paper_title") or "Unknown title"
    link = metadata.get("link") or metadata.get("url") or ""
    authors = metadata.get("authors") or metadata.get("author") or []
    if isinstance(authors, list):
        authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    else:
        authors_str = str(authors)
    year = ""
    if metadata.get("published"):
        try:
            year = str(metadata.get("published"))[:4]
        except Exception:
            year = ""
    parts = [authors_str, f"\"{title}\""]
    if year:
        parts.append(year)
    if link:
        parts.append(link)
    return ", ".join([p for p in parts if p])
