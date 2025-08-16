# src/index/build_index.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.settings import settings
from src.ingest.fetch_arxiv import fetch_arxiv_papers
from src.ingest.normalize import normalize_text, run_normalization
from src.ingest.chunk import chunk_text, run_chunking

# Define constants for the collection name and file paths.
COLLECTION_NAME = "physics_papers"
DATA_CSV = settings.CHUNK_OUTPUT_PATH
CHROMA_DIR = settings.CHROMA_DIR

def get_chroma_client(persist=True):
    """
    Create a Chroma client. For environments that can't use sqlite/PersistentClient,
    set persist=False to try an in-memory client.
    This function handles different client initialization methods for compatibility.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        # Raise an error if the chromadb library is not installed.
        raise RuntimeError("chromadb import failed: " + str(e))

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Some chromadb installs (older/newer) expose different constructors.
    # Try to create a persistent client, otherwise fall back to simple Client.
    try:
        # Attempt to create a persistent client with a specified directory.
        return chromadb.Client(Settings(persist_directory=CHROMA_DIR, anonymized_telemetry=False))
    except Exception:
        try:
            # If that fails, try a simpler, non-persistent client.
            return chromadb.Client(Settings(anonymized_telemetry=False))
        except Exception as e:
            # last resort: try in-memory (may still fail)
            return chromadb.Client()

def build_index_from_csv(client=None):
    """Build the ChromaDB index from a pre-chunked CSV file."""
    client = client or get_chroma_client()
    # Get or create the ChromaDB collection.
    coll = client.get_or_create_collection(COLLECTION_NAME)

    if not os.path.exists(DATA_CSV):
        # Raise an error if the chunked data CSV is missing.
        raise FileNotFoundError(f"{DATA_CSV} not found. Run ingestion & chunking first.")

    # Read the data from the CSV file.
    df = pd.read_csv(DATA_CSV)
    if "chunk" not in df.columns:
        # Ensure the required 'chunk' column exists.
        raise ValueError("CSV must have 'chunk' column.")

    # Prepare data for indexing.
    ids = [str(i) for i in range(len(df))]
    docs = df["chunk"].astype(str).tolist()
    metas = df.to_dict(orient="records")

    # Load the embedding model.
    model = SentenceTransformer(settings.EMBED_MODEL)
    # Encode the documents into vector embeddings.
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()

    # Add the data to the ChromaDB collection.
    coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"✅ Index built from CSV with {len(docs)} documents.")
    return coll

def build_index_from_scratch(query=None, max_results=None, client=None):
    """Build the index by running the full ingestion pipeline from fetching papers."""
    client = client or get_chroma_client()
    coll = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(settings.EMBED_MODEL)

    query = query or settings.ARXIV_QUERY
    max_results = max_results or settings.ARXIV_MAX_RESULTS
    # Fetch papers directly from arXiv.
    papers = fetch_arxiv_papers(query=query, max_results=max_results, output_file=settings.RAW_DATA_PATH)

    docs, metas = [], []
    for paper in papers:
        # Normalize and chunk each paper's summary.
        normalized = normalize_text(paper["summary"])
        for chunk in chunk_text(normalized):
            docs.append(chunk)
            # Store metadata for each chunk.
            metas.append({
                "title": paper["title"],
                "url": paper["link"],
                "published": paper.get("published"),
                "authors": paper.get("authors")
            })

    ids = [f"doc_{i}" for i in range(len(docs))]
    # Encode all the new chunks.
    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    # Add the chunks to the index.
    coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"✅ Index built from scratch with {len(docs)} chunks.")
    return coll

def ensure_index_exists(client=None):
    """
    Checks if a ChromaDB index exists. If not, it runs the necessary pipeline
    to build it either from a CSV or from scratch.
    """
    client = client or get_chroma_client()
    try:
        # Try to get an existing collection.
        coll = client.get_collection(COLLECTION_NAME)
        if coll.count() > 0:
            print(f"ℹ️ Using existing collection '{COLLECTION_NAME}' with {coll.count()} docs.")
            return coll
    except Exception:
        print("⚠️ No existing collection found. Will build index now.")

    # Build from CSV if available, else run full pipeline.
    if os.path.exists(DATA_CSV):
        # Build from the pre-chunked CSV.
        coll = build_index_from_csv(client=client)
    else:
        # Run the full ingestion pipeline: fetch -> normalize -> chunk -> build.
        fetch_arxiv_papers(output_file=settings.RAW_DATA_PATH, query=settings.ARXIV_QUERY, max_results=settings.ARXIV_MAX_RESULTS)
        run_normalization(input_file=settings.RAW_DATA_PATH, output_file=settings.CLEAN_DATA_PATH)
        run_chunking(input_file=settings.CLEAN_DATA_PATH, output_file=settings.CHUNK_OUTPUT_PATH)
        coll = build_index_from_csv(client=client)

    return coll

if __name__ == "__main__":
    # This block allows the file to be executed as a script to build the index.
    client = get_chroma_client()
    if os.path.exists(DATA_CSV):
        build_index_from_csv(client=client)
    else:
        build_index_from_scratch(client=client)
