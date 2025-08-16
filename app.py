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
    # Check if the punkt tokenizer is already downloaded and available.
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        # If not, attempt to download it quietly.
        nltk.download("punkt", quiet=True)
    except Exception:
        # This handles cases where the download might be restricted (e.g., in a cloud environment).
        # The normalization/chunking modules will attempt to download it again locally.
        pass

# Cache the index creation so repeated Streamlit reruns don't rebuild repeatedly
@st.cache_resource(show_spinner="Building or loading the document index...")
def create_or_get_collection():
    """
    Create a Chroma client, run ingestion pipeline if necessary, and return the collection.
    This function is decorated with `@st.cache_resource` to ensure the expensive
    index creation process runs only once per session.
    """
    # Create Chroma client (function lives in src.index.build_index)
    try:
        # Initialize the ChromaDB client to manage the vector database.
        client = get_chroma_client()
    except Exception as e:
        # Display an error message if the client fails to initialize.
        st.error("Failed to initialize ChromaDB client. Check chromadb install and sqlite version.")
        st.error(str(e))
        raise

    # If chunk CSV doesn't exist, run ingestion pipeline then build index
    try:
        # The main logic to check if the data has already been processed and indexed.
        if not os.path.exists(settings.CHUNK_OUTPUT_PATH):
            #st.info("No chunked data found — running ingestion pipeline (fetch → normalize → chunk → index).")

            # 1) Fetch (will write settings.RAW_DATA_PATH)
            st.write("1️⃣ Fetching papers from arXiv...")
            # Calls the function to fetch physics papers from arXiv based on settings.
            fetch_arxiv_papers(output_file=settings.RAW_DATA_PATH,
                               query=settings.ARXIV_QUERY,
                               max_results=settings.ARXIV_MAX_RESULTS)

            # 2) Normalize (writes CLEAN_DATA_PATH)
            st.write("2️⃣ Normalizing text and extracting equations...")
            # Cleans the fetched text and extracts equations.
            run_normalization(input_file=settings.RAW_DATA_PATH,
                              output_file=settings.CLEAN_DATA_PATH)

            # 3) Chunk (writes CHUNK_OUTPUT_PATH)
            st.write("3️⃣ Chunking documents...")
            # Splits the normalized text into smaller, manageable chunks.
            run_chunking(input_file=settings.CLEAN_DATA_PATH,
                         output_file=settings.CHUNK_OUTPUT_PATH)

            # 4) Build index from CSV into the provided client
            st.write("4️⃣ Building vector index from chunks...")
            # Creates the ChromaDB vector index from the generated chunks.
            build_index_from_csv(client=client)

        # Ensure the collection exists and return it
        # This function either gets the existing collection or creates a new one.
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
    # Attempt to get or create the document collection.
    collection = create_or_get_collection()
except Exception:
    # If the process fails, stop the Streamlit app to prevent further errors.
    st.stop()

# --- UI ---
# Set up the main user interface for the RAG application.
st.header("Ask a question about physics papers")
# Text input box for the user to type their question.
query = st.text_input("Enter your physics question here:", placeholder="e.g., What is Hawking radiation?")

if st.button("Generate Answer"):
    # Logic that runs when the "Generate Answer" button is clicked.
    if not query or not query.strip():
        # Display a warning if the user hasn't entered a question.
        st.warning("Please enter a question first.")
    else:
        # Show a spinner while the RAG process is running.
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # generate_answer will retrieve top_k hits and use top 4 as LLM context (per generator design)
                # Call the main function to generate the answer using the RAG pipeline.
                res = generate_answer(query=query, collection=collection, top_k=settings.TOP_K, max_length=300, temperature=0.0)

                # Display answer
                st.markdown("### 🧠 Answer")
                # Show the generated answer or a fallback message.
                st.write(res.get("answer", "No answer returned."))

                # Display citations (res["hits"] is expected to be list of dicts with 'reference')
                st.markdown("### 📚 References")
                hits = res.get("hits", [])
                if not hits:
                    # Inform the user if no citations were found.
                    st.write("No citations returned.")
                else:
                    # Iterate through the retrieved hits to display citations.
                    for i, h in enumerate(hits, 1):
                        # Support either structured dict with 'reference' or fallback formatting
                        if isinstance(h, dict) and "reference" in h:
                            # Display the pre-formatted citation string.
                            st.markdown(f"- {h['reference']}")
                        elif isinstance(h, dict) and "metadata" in h:
                            # fallback: format from metadata if a pre-formatted reference isn't available.
                            meta = h.get("metadata", {})
                            title = meta.get("title", "Untitled")
                            url = meta.get("url") or meta.get("link") or ""
                            # Display the citation with a hyperlink if a URL exists.
                            st.markdown(f"- [{title}]({url})" if url else f"- {title}")
                        else:
                            # Final fallback for any other unexpected hit format.
                            st.markdown(f"- {h}")

                # Display equation checks if any
                eq_checks = res.get("equation_checks", [])
                if eq_checks:
                    st.markdown("### ➗ Extracted / Verified Equations")
                    # Iterate through and display the results of the equation verification.
                    for chk in eq_checks:
                        expr = chk.get("expr", "")
                        status = chk.get("status", "")
                        notes = chk.get("notes", "")
                        # Format and display the equation, its status, and any parsing notes.
                        st.markdown(f"- `${expr}$` — **{status}** — {notes}")

            except Exception as e:
                # Handle and display any errors that occur during the generation process.
                st.error("Generation failed. See error below.")
                st.error(str(e))
                st.text(traceback.format_exc())

st.markdown("---")
# Informational footer about the application's backend flow.
#st.info("Backend: arXiv → normalize → chunk → ChromaDB → LLM (RAG).")
