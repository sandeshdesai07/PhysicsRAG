# src/ingest/fetch_arxiv.py
import arxiv
import pandas as pd
import os
import requests
import fitz  # PyMuPDF
from src.settings import settings


def fetch_arxiv_papers(query=None, max_results=None, output_file=None):
    """
    Fetch papers from arXiv (with full PDF text) and save to CSV.
    Args:
        query (str): Search query for arXiv. Defaults to settings.ARXIV_QUERY.
        max_results (int): Maximum number of papers to fetch. Defaults to settings.ARXIV_MAX_RESULTS.
        output_file (str): CSV file path to save the results.
        Defaults to 'arxiv_data.csv' in repo root.

    Returns:
        list[dict]: List of paper dictionaries.
    """
    # Use provided arguments or fall back to default settings.
    query = query or settings.ARXIV_QUERY
    max_results = max_results or settings.ARXIV_MAX_RESULTS
    output_file = output_file or os.path.join(os.path.dirname(__file__), "../../arxiv_data.csv")

    papers = []

    # Configure the arXiv search.
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    # Iterate through the search results.
    for result in search.results():
        print(f"📄 Fetching: {result.title}")
        pdf_text = ""
        try:
            # Attempt to download the PDF.
            pdf_url = result.pdf_url
            response = requests.get(pdf_url, timeout=20)
            pdf_path = "temp_arxiv.pdf"

            with open(pdf_path, "wb") as f:
                f.write(response.content)

            # Extract text from PDF using PyMuPDF.
            with fitz.open(pdf_path) as doc:
                pdf_text = "\n".join(page.get_text() for page in doc)

            # Clean up the temporary PDF file.
            os.remove(pdf_path)

        except Exception as e:
            print(f"⚠️ Failed to fetch PDF for {result.title}: {e}")

        # Append the paper's metadata and extracted text to the list.
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "full_text": pdf_text,
            "link": result.entry_id,
            "published": result.published,
            "authors": [author.name for author in result.authors]
        })

    # Convert the list of papers to a pandas DataFrame and save to CSV.
    df = pd.DataFrame(papers)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Fetched {len(df)} papers (with PDFs) and saved to {output_file}")
    return papers


if __name__ == "__main__":
    # Allow the script to be run directly for testing.
    fetch_arxiv_papers()
