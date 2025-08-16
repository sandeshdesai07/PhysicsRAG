# src/demo/quick_query.py
import re
from sentence_transformers import SentenceTransformer

# Load embedding model globally
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_chroma_client():
    """Return in-memory ChromaDB client (avoids SQLite version issues)."""
    import chromadb
    from chromadb.config import Settings
    return chromadb.Client(Settings(anonymized_telemetry=False))


def answer_query(query, collection=None, top_k=3):
    """
    Search collection and return an answer with citations + equations.
    """
    if not query.strip():
        return {"answer": "No query provided.", "citations": [], "equations": []}

    if collection is None:
        client = get_chroma_client()
        try:
            collection = client.get_collection("physics_papers")
        except Exception:
            return {"answer": "No index available.", "citations": [], "equations": []}

    # Encode query
    embedding = MODEL.encode(query).tolist()

    # Retrieve docs
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        return {"answer": "No relevant documents found.", "citations": [], "equations": []}

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Generate answer snippet
    answer_text = "Based on the retrieved documents:\n\n"
    for doc in docs:
        snippet = doc.strip().replace("\n", " ")
        answer_text += f"- {snippet[:200]}...\n"

    # Extract LaTeX equations
    equations = []
    for doc in docs:
        equations.extend(re.findall(r"\$.*?\$", doc))

    citations = [{"title": m.get("title", "Untitled"), "url": m.get("url", "")} for m in metas]

    return {"answer": answer_text.strip(), "citations": citations, "equations": equations}

# Optional CLI test
if __name__ == "__main__":
    from src.index.build_index import ensure_index_exists
    coll = ensure_index_exists()
    q = input("Enter your physics question: ")
    res = answer_query(q, coll)
    print("\nANSWER:\n", res["answer"])
    print("\nCITATIONS:")
    for c in res["citations"]:
        print(f"- {c['title']} → {c['url']}")
