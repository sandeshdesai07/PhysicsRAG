# src/index/build_index.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.settings import settings
from src.ingest.fetch_arxiv import fetch_arxiv_papers
from src.ingest.normalize import normalize_text, run_normalization
from src.ingest.chunk import chunk_text, run_chunking

COLLECTION_NAME = "physics_papers"
DATA_CSV = settings.CHUNK_OUTPUT_PATH
CHROMA_DIR = settings.CHROMA_DIR

def get_chroma_client(persist=True):
    """
    Create a Chroma client. For environments that can't use sqlite/PersistentClient,
    set persist=False to try an in-memory client.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        raise RuntimeError("chromadb import failed: " + str(e))

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Some chromadb installs (older/newer) expose different constructors.
    # Try to create a persistent client, otherwise fall back to simple Client.
    try:
        return chromadb.Client(Settings(persist_directory=CHROMA_DIR, anonymized_telemetry=False))
    except Exception:
        try:
            # older or different interface
            return chromadb.Client(Settings(anonymized_telemetry=False))
        except Exception as e:
            # last resort: try in-memory (may still fail)
            return chromadb.Client()

def build_index_from_csv(client=None):
    client = client or get_chroma_client()
    coll = client.get_or_create_collection(COLLECTION_NAME)

    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Run ingestion & chunking first.")

    df = pd.read_csv(DATA_CSV)
    if "chunk" not in df.columns:
        raise ValueError("CSV must have 'chunk' column.")

    ids = [str(i) for i in range(len(df))]
    docs = df["chunk"].astype(str).tolist()
    metas = df.to_dict(orient="records")

    model = SentenceTransformer(settings.EMBED_MODEL)
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()

    coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"✅ Index built from CSV with {len(docs)} documents.")
    return coll

def build_index_from_scratch(query=None, max_results=None, client=None):
    client = client or get_chroma_client()
    coll = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(settings.EMBED_MODEL)

    query = query or settings.ARXIV_QUERY
    max_results = max_results or settings.ARXIV_MAX_RESULTS
    papers = fetch_arxiv_papers(query=query, max_results=max_results, output_file=settings.RAW_DATA_PATH)

    docs, metas = [], []
    for paper in papers:
        normalized = normalize_text(paper["summary"])
        for chunk in chunk_text(normalized):
            docs.append(chunk)
            metas.append({
                "title": paper["title"],
                "url": paper["link"],
                "published": paper.get("published"),
                "authors": paper.get("authors")
            })

    ids = [f"doc_{i}" for i in range(len(docs))]
    embeddings = model.encode(docs, show_progress_bar=True).tolist()
    coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"✅ Index built from scratch with {len(docs)} chunks.")
    return coll

def ensure_index_exists(client=None):
    client = client or get_chroma_client()
    try:
        coll = client.get_collection(COLLECTION_NAME)
        if coll.count() > 0:
            print(f"ℹ️ Using existing collection '{COLLECTION_NAME}' with {coll.count()} docs.")
            return coll
    except Exception:
        print("⚠️ No existing collection found. Will build index now.")

    # Build from CSV if available, else run full pipeline
    if os.path.exists(DATA_CSV):
        coll = build_index_from_csv(client=client)
    else:
        # fetch -> normalize -> chunk -> build
        fetch_arxiv_papers(output_file=settings.RAW_DATA_PATH, query=settings.ARXIV_QUERY, max_results=settings.ARXIV_MAX_RESULTS)
        run_normalization(input_file=settings.RAW_DATA_PATH, output_file=settings.CLEAN_DATA_PATH)
        run_chunking(input_file=settings.CLEAN_DATA_PATH, output_file=settings.CHUNK_OUTPUT_PATH)
        coll = build_index_from_csv(client=client)

    return coll

if __name__ == "__main__":
    client = get_chroma_client()
    if os.path.exists(DATA_CSV):
        build_index_from_csv(client=client)
    else:
        build_index_from_scratch(client=client)
