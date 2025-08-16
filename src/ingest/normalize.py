# src/ingest/normalize.py
import os
import pandas as pd
import re
import nltk
import fitz  # PyMuPDF
from src.ingest.fetch_arxiv import fetch_arxiv_papers
from src.settings import settings

# Ensure a persistent local nltk_data folder inside your repo
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download punkt if not found
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path, quiet=True)


def normalize_text(text):
    """Clean and normalize whitespace in text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_equations(text):
    """Extract LaTeX-style equations from text."""
    if not isinstance(text, str):
        return []
    eq_pattern = r"\$.*?\$"
    return re.findall(eq_pattern, text)


def extract_pdf_text(pdf_path):
    """Extract text from PDF if it exists."""
    if pdf_path and os.path.exists(pdf_path):
        try:
            with fitz.open(pdf_path) as pdf:
                return "\n".join(page.get_text() for page in pdf)
        except Exception as e:
            print(f"⚠️ PDF extraction failed for {pdf_path}: {e}")
    return ""


def run_normalization(input_file=None, output_file=None):
    input_file = input_file or settings.RAW_DATA_PATH
    output_file = output_file or settings.CLEAN_DATA_PATH

    # If input file missing, fetch papers from arXiv first
    if not os.path.exists(input_file):
        print(f"⚠️ {input_file} not found. Fetching arXiv papers first...")
        fetch_arxiv_papers(
            output_file=input_file,
            query=settings.ARXIV_QUERY,
            max_results=settings.ARXIV_MAX_RESULTS
        )

    df = pd.read_csv(input_file)

    # Ensure we have expected columns
    if "summary" not in df.columns:
        df["summary"] = ""
    if "pdf_path" not in df.columns:
        df["pdf_path"] = ""

    # Combine PDF text or abstract as `content`
    contents = []
    for _, row in df.iterrows():
        pdf_text = extract_pdf_text(row["pdf_path"])
        if pdf_text.strip():
            contents.append(pdf_text)
        else:
            contents.append(row["summary"])
    df["content"] = contents

    # Normalize & extract equations
    df["clean_content"] = df["content"].apply(normalize_text)
    df["equations"] = df["content"].apply(extract_equations)

    # Remove completely empty entries
    df = df[df["clean_content"].str.strip() != ""]

    df.to_csv(output_file, index=False)
    print(f"✅ Normalized text (PDF or abstract) and extracted equations → {output_file}")


if __name__ == "__main__":
    run_normalization()
