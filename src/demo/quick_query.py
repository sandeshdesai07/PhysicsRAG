# src/demo/quick_query.py
import re
from sentence_transformers import SentenceTransformer

# Load embedding model globally
# The embedding model is loaded once to avoid repeated, slow loading.
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_chroma_client():
    """
    Return in-memory ChromaDB client (avoids SQLite version issues).
    This function provides a simple, in-memory client for testing purposes.
    """
    import chromadb
    from chromadb.config import Settings
    return chromadb.Client(Settings(anonymized_telemetry=False))


def answer_query(query, collection=None, top_k=3):
    """
    Search collection and return an answer with citations + equations.
    This function performs the retrieval and basic answer generation.
    """
    if not query.strip():
        # Handle empty queries gracefully.
        return {"answer": "No query provided.", "citations": [], "equations": []}

    if collection is None:
        # If no collection is provided, try to get an existing one.
        client = get_chroma_client()
        try:
            collection = client.get_collection("physics_papers")
        except Exception:
            # Handle the case where the index doesn't exist.
            return {"answer": "No index available.", "citations": [], "equations": []}

    # Encode query
    # Convert the user's query into a vector embedding.
    embedding = MODEL.encode(query).tolist()

    # Retrieve docs
    # Query the vector database to find the most relevant documents.
    results = collection.query(query_embeddings=[embedding], n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        # Handle the case where no relevant documents were found.
        return {"answer": "No relevant documents found.", "citations": [], "equations": []}

    # Extract the retrieved documents and their metadata.
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # Generate answer snippet
    answer_text = "Based on the retrieved documents:\n\n"
    # Construct a simple answer by concatenating snippets from the retrieved documents.
    for doc in docs:
        snippet = doc.strip().replace("\n", " ")
        answer_text += f"- {snippet[:200]}...\n"

    # Extract LaTeX equations
    equations = []
    # Use a regular expression to find all inline LaTeX equations.
    for doc in docs:
        equations.extend(re.findall(r"\$.*?\$", doc))

    # Format the citations from the retrieved metadata.
    citations = [{"title": m.get("title", "Untitled"), "url": m.get("url", "")} for m in metas]

    # Return the generated answer, citations, and equations.
    return {"answer": answer_text.strip(), "citations": citations, "equations": equations}

# Optional CLI test
if __name__ == "__main__":
    # This block allows the script to be run directly from the command line for testing.
    from src.index.build_index import ensure_index_exists
    # Ensure the index exists before querying.
    coll = ensure_index_exists()
    q = input("Enter your physics question: ")
    res = answer_query(q, coll)
    print("\nANSWER:\n", res["answer"])
    print("\nCITATIONS:")
    for c in res["citations"]:
        print(f"- {c['title']} → {c['url']}")
