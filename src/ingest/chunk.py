# src/ingest/chunk.py
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# persistent nltk_data folder at repo root
nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Download required NLTK resources if not present
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)

def _simple_normalize(text):
    """Very small fallback normalizer if clean_summary was not produced."""
    if not isinstance(text, str):
        return ""
    # collapse whitespace, remove repeated newlines
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=500):
    """Split text into word chunks (approx chunk_size words)."""
    if not isinstance(text, str) or not text.strip():
        return []
    words = word_tokenize(text)
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def run_chunking(input_file="arxiv_clean.csv", output_file="data_chunks.csv"):
    """
    Read normalized summaries from input_file, split into chunks, and save to output_file.
    If 'clean_summary' column is missing but 'summary' exists, use 'summary' with basic normalization.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found. Please run normalization first.")

    df = pd.read_csv(input_file)

    # Diagnostic: list available columns
    cols = df.columns.tolist()
    print(f"🔎 run_chunking: found columns: {cols}")

    # Ensure we have a text column to chunk
    if "clean_summary" not in df.columns:
        if "summary" in df.columns:
            print("⚠️ 'clean_summary' missing — falling back to 'summary' and applying basic normalization.")
            df["clean_summary"] = df["summary"].astype(str).apply(_simple_normalize)
        else:
            # helpful error message for debugging
            raise ValueError(
                "Input CSV must contain 'clean_summary' or 'summary' column. "
                f"Found columns: {cols}"
            )

    # Validate required metadata columns exist; if missing, add empty placeholders
    if "title" not in df.columns:
        print("⚠️ 'title' column missing — filling with 'Untitled'.")
        df["title"] = "Untitled"
    if "link" not in df.columns:
        print("⚠️ 'link' column missing — filling with empty string.")
        df["link"] = ""

    chunks = []
    for _, row in df.iterrows():
        text = row["clean_summary"]
        for c in chunk_text(text):
            chunks.append({
                "title": row.get("title", "Untitled"),
                "chunk": c,
                "link": row.get("link", "")
            })

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_csv(output_file, index=False)
    print(f"✅ Chunked text → {output_file} ({len(chunks)} chunks)")

if __name__ == "__main__":
    run_chunking()
