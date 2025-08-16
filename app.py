# app.py
import os
import streamlit as st
import nltk
import traceback

from src.settings import settings
from src.ingest.fetch_arxiv import fetch_arxiv_papers
from src.ingest.normalize import run_normalization
from src.ingest.chunk import run_chunking
from src.index.build_index import get_chroma_client, build_index_from_csv, ensure_index_exists
from src.generator.generate import generate_answer

# Page config
st.set_page_config(page_title="PhysiCalc-RAG", layout="wide")
st.title("📘 PhysiCalc-RAG: Physics Research Assistant")

# Ensure NLTK punkt tokenizer is available (best-effort; modules also try to download)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        # If download isn't allowed (cloud), normalization/chunk modules try again locally.
        pass

# Cache the index creation so repeated Streamlit reruns don't rebuild repeatedly
@st.cache_resource(show_spinner="Building or loading the document index...")
def create_or_get_collection():
    """
    Create a Chroma client, run ingestion pipeline if necessary, and return the collection.
    """
    # Create Chroma client (function lives in src.index.build_index)
    try:
        client = get_chroma_client()
    except Exception as e:
        st.error("Failed to initialize ChromaDB client. Check chromadb install and sqlite version.")
        st.error(str(e))
        raise

    # If chunk CSV doesn't exist, run ingestion pipeline then build index
    try:
        if not os.path.exists(settings.CHUNK_OUTPUT_PATH):
            st.info("No chunked data found — running ingestion pipeline (fetch → normalize → chunk → index).")

            # 1) Fetch (will write settings.RAW_DATA_PATH)
            st.write("1️⃣ Fetching papers from arXiv...")
            fetch_arxiv_papers(output_file=settings.RAW_DATA_PATH,
                               query=settings.ARXIV_QUERY,
                               max_results=settings.ARXIV_MAX_RESULTS)

            # 2) Normalize (writes CLEAN_DATA_PATH)
            st.write("2️⃣ Normalizing text and extracting equations...")
            run_normalization(input_file=settings.RAW_DATA_PATH,
                              output_file=settings.CLEAN_DATA_PATH)

            # 3) Chunk (writes CHUNK_OUTPUT_PATH)
            st.write("3️⃣ Chunking documents...")
            run_chunking(input_file=settings.CLEAN_DATA_PATH,
                        output_file=settings.CHUNK_OUTPUT_PATH)

            # 4) Build index from CSV into the provided client
            st.write("4️⃣ Building vector index from chunks...")
            build_index_from_csv(client=client)

        # Ensure the collection exists and return it
        collection = ensure_index_exists(client=client)
        return collection

    except Exception as e:
        # Bubble up so Streamlit shows an error and caches the failure
        st.error("Error while creating/ensuring index. See logs for details.")
        st.error(str(e))
        st.text(traceback.format_exc())
        raise

# Initialize collection (will build if needed)
try:
    collection = create_or_get_collection()
except Exception:
    st.stop()

# --- UI ---
st.header("Ask a question about physics papers")
query = st.text_input("Enter your physics question here:", placeholder="e.g., What is Hawking radiation?")

if st.button("Generate Answer"):
    if not query or not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # generate_answer will retrieve top_k hits and use top 4 as LLM context (per generator design)
                res = generate_answer(query=query, collection=collection, top_k=settings.TOP_K, max_length=300, temperature=0.0)

                # Display answer
                st.markdown("### 🧠 Answer")
                st.write(res.get("answer", "No answer returned."))

                # Display citations (res["hits"] is expected to be list of dicts with 'reference')
                st.markdown("### 📚 References")
                hits = res.get("hits", [])
                if not hits:
                    st.write("No citations returned.")
                else:
                    for i, h in enumerate(hits, 1):
                        # Support either structured dict with 'reference' or fallback formatting
                        if isinstance(h, dict) and "reference" in h:
                            st.markdown(f"- {h['reference']}")
                        elif isinstance(h, dict) and "metadata" in h:
                            # fallback: format from metadata
                            meta = h.get("metadata", {})
                            title = meta.get("title", "Untitled")
                            url = meta.get("url") or meta.get("link") or ""
                            st.markdown(f"- [{title}]({url})" if url else f"- {title}")
                        else:
                            st.markdown(f"- {h}")

                # Display equation checks if any
                eq_checks = res.get("equation_checks", [])
                if eq_checks:
                    st.markdown("### ➗ Extracted / Verified Equations")
                    for chk in eq_checks:
                        expr = chk.get("expr", "")
                        status = chk.get("status", "")
                        notes = chk.get("notes", "")
                        st.markdown(f"- `${expr}$` — **{status}** — {notes}")

            except Exception as e:
                st.error("Generation failed. See error below.")
                st.error(str(e))
                st.text(traceback.format_exc())

st.markdown("---")
st.info("Backend: arXiv → normalize → chunk → ChromaDB → LLM (RAG).")
