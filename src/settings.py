# src/settings.py
from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEN_MODEL: str = "google/flan-t5-small"
    ARXIV_QUERY: str = "physics"
    ARXIV_MAX_RESULTS: int = 200

    # file paths (consistent)
    RAW_DATA_PATH: str = os.path.join("src", "ingest", "arxiv_data.csv")
    CLEAN_DATA_PATH: str = os.path.join("src", "ingest", "arxiv_clean.csv")
    CHUNK_OUTPUT_PATH: str = os.path.join("src", "ingest", "data_chunks.csv")

    # chroma
    CHROMA_DIR: str = os.path.join("chromadb_data")

    TOP_K: int = 5

settings = Settings()
