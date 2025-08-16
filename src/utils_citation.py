# src/utils_citation.py
from typing import Dict, Any

def format_ieee(metadata: Dict[str, Any]) -> str:
    """
    Build a short IEEE-like citation string from metadata produced in indexing.
    This function takes a metadata dictionary and formats it into a human-readable
    citation string, similar to the IEEE style.
    
    metadata expected keys: title, link, published, authors (maybe list)
    """
    # Safely retrieve the title with a fallback.
    title = metadata.get("title") or metadata.get("paper_title") or "Unknown title"
    # Safely retrieve the link/URL.
    link = metadata.get("link") or metadata.get("url") or ""
    # Safely retrieve authors, handling both list and string formats.
    authors = metadata.get("authors") or metadata.get("author") or []
    if isinstance(authors, list):
        # Format the authors list, showing up to the first three and adding "et al." if more exist.
        authors_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    else:
        authors_str = str(authors)
    
    # Safely extract the publication year.
    year = ""
    if metadata.get("published"):
        try:
            # Get the first 4 characters of the 'published' field, assuming it's a date string.
            year = str(metadata.get("published"))[:4]
        except Exception:
            year = ""
    
    # Assemble the parts of the citation.
    parts = [authors_str, f"\"{title}\""]
    if year:
        parts.append(year)
    if link:
        parts.append(link)
    
    # Join the non-empty parts with a comma and space.
    return ", ".join([p for p in parts if p])
