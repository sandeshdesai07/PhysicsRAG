# src/settings.py
from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    """
    Centralized configuration management for the RAG application.
    This class uses Pydantic to handle settings with defaults.
    """
    # Model settings
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Model for creating embeddings
    GEN_MODEL: str = "google/flan-t5-small"                       # Model for text generation (LLM)

    # arXiv ingestion settings
    ARXIV_QUERY: str = "physics"           # Default search query for arXiv papers
    ARXIV_MAX_RESULTS: int = 200            # Maximum number of papers to fetch

    # file paths (consistent)
    # Defines the file paths for the raw, cleaned, and chunked data.
    RAW_DATA_PATH: str = os.path.join("src", "ingest", "arxiv_data.csv")
    CLEAN_DATA_PATH: str = os.path.join("src", "ingest", "arxiv_clean.csv")
    CHUNK_OUTPUT_PATH: str = os.path.join("src", "ingest", "data_chunks.csv")

    # chroma settings
    CHROMA_DIR: str = os.path.join("chromadb_data")  # Directory to store the persistent ChromaDB index

    # RAG pipeline settings
    TOP_K: int = 5  # Number of top documents to retrieve from the index

# Create a single instance of the settings to be imported across modules.
settings = Settings()
