# PhysiCalc-RAG — Phase 2 (Streamlit RAG demo)

## What this contains
- Phase 1 indexing pipeline (ingest → normalize → chunk → build_index)
- Phase 2 Streamlit app (`app.py`) that does retrieval-augmented generation (RAG)
  using a free generation model (`google/flan-t5-small`), ChromaDB, and Sentence-Transformers.
- SymPy-based equation parsing/verification.

## How to deploy to GitHub & Hugging Face Spaces (no local push required)
1. Create a new GitHub repository and upload all files (you can use GitHub web UI to add files).
2. Ensure CSV files from Phase 1 (`data_arxiv.csv`, `data_arxiv_clean.csv`, `data_chunks.csv`)
   are present in repo `root`. If not present, run Phase 1 (or run the Colab notebook to generate them
   then upload).
3. Go to https://huggingface.co/spaces, click "Create new Space".
4. Choose "Streamlit" as SDK, link it to your GitHub repo (enter repo name) or upload the repo directly.
5. For `Hardware` choose CPU (free). If your app needs GPU inference later you can switch.
6. Wait for Space to build. The app will run `app.py` on start; if index is missing the app will build it from CSV.

## Local testing
1. Create Python venv, install `pip install -r requirements.txt`.
2. `cp .env.example .env` and edit if desired.
3. Run:
