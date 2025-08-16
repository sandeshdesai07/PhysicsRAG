# src/ingest/chunk.py
import os
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# persistent nltk_data folder at repo root
# Configure NLTK to use a local data directory.
nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Download required NLTK resources if not present
# Ensure the 'punkt' and 'punkt_tab' tokenizers are downloaded.
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path, quiet=True)

def _simple_normalize(text):
    """
    Very small fallback normalizer if clean_summary was not produced.
    This function handles basic whitespace cleanup.
    """
    if not isinstance(text, str):
        return ""
    # Collapse multiple whitespace characters into a single space and strip leading/trailing space.
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=500):
    """Split text into word chunks (approx chunk_size words)."""
    if not isinstance(text, str) or not text.strip():
        # Return an empty list for invalid input.
        return []
    # Tokenize the text into a list of words.
    words = word_tokenize(text)
    # Split the list of words into chunks of `chunk_size`.
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def run_chunking(input_file="arxiv_clean.csv", output_file="data_chunks.csv"):
    """
    Read normalized summaries from input_file, split into chunks, and save to output_file.
    If 'clean_summary' column is missing but 'summary' exists, use 'summary' with basic normalization.
    This function orchestrates the chunking process for all documents.
    """
    if not os.path.exists(input_file):
        # Raise an error if the input file from the normalization step is missing.
        raise FileNotFoundError(f"{input_file} not found. Please run normalization first.")

    # Read the input CSV file into a pandas DataFrame.
    df = pd.read_csv(input_file)

    # Diagnostic: list available columns
    cols = df.columns.tolist()
    print(f"🔎 run_chunking: found columns: {cols}")

    # Ensure we have a text column to chunk
    if "clean_summary" not in df.columns:
        if "summary" in df.columns:
            print("⚠️ 'clean_summary' missing — falling back to 'summary' and applying basic normalization.")
            # Use the 'summary' column if 'clean_summary' is not present.
            df["clean_summary"] = df["summary"].astype(str).apply(_simple_normalize)
        else:
            # helpful error message for debugging
            raise ValueError(
                "Input CSV must contain 'clean_summary' or 'summary' column. "
                f"Found columns: {cols}"
            )

    # Validate required metadata columns exist; if missing, add empty placeholders
    # These columns are needed for creating the citations later.
    if "title" not in df.columns:
        print("⚠️ 'title' column missing — filling with 'Untitled'.")
        df["title"] = "Untitled"
    if "link" not in df.columns:
        print("⚠️ 'link' column missing — filling with empty string.")
        df["link"] = ""

    chunks = []
    # Iterate through each row of the DataFrame to process the text.
    for _, row in df.iterrows():
        text = row["clean_summary"]
        # Call the chunking function for each document.
        for c in chunk_text(text):
            # Append a dictionary for each chunk, including its metadata.
            chunks.append({
                "title": row.get("title", "Untitled"),
                "chunk": c,
                "link": row.get("link", "")
            })

    # Convert the list of chunk dictionaries into a new DataFrame.
    chunks_df = pd.DataFrame(chunks)
    # Save the chunked data to a new CSV file.
    chunks_df.to_csv(output_file, index=False)
    print(f"✅ Chunked text → {output_file} ({len(chunks)} chunks)")

if __name__ == "__main__":
    # Allow the script to be run directly for testing.
    run_chunking()
