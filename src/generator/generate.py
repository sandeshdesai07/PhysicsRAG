# src/generator/generate.py
import os
import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

from sympy.parsing.latex import parse_latex
from sympy import SympifyError

from src.settings import settings
from src.utils_citation import format_ieee

# Ensure punkt is available for sentence tokenization (safe no-op if present)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Globals
GEN_MODEL = settings.GEN_MODEL
EMBED_MODEL = settings.EMBED_MODEL

_tokenizer = None
_model = None
_gen_pipeline = None
_embed_model = None

def _init_models():
    """Initialize models only once for efficiency. Generation model may be large."""
    global _tokenizer, _model, _gen_pipeline, _embed_model
    if _gen_pipeline is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
            _model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
            _gen_pipeline = pipeline(
                "text2text-generation",
                model=_model,
                tokenizer=_tokenizer,
                device=-1  # CPU by default; change to 0 for GPU if available
            )
        except Exception as e:
            # Generation model not available — we'll fall back to extractive summarizer below.
            _gen_pipeline = None
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _gen_pipeline, _embed_model

def retrieve_contexts(collection, query: str, top_k: int):
    """Retrieve top_k relevant docs from the Chroma collection and return a list of hits."""
    _, embed_model = _init_models()
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    docs_list = res.get("documents", [[]])[0]
    metas_list = res.get("metadatas", [[]])[0]
    dists_list = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs_list, metas_list, dists_list):
        hits.append({"document": doc, "metadata": meta or {}, "distance": dist})
    return hits

def build_prompt(query: str, hits: List[Dict]):
    """Construct a prompt for the LLM from up to 4 retrieved documents."""
    header = (
        "You are a helpful physics assistant. Use ONLY the information in the provided snippets "
        "to answer the question. Be concise and provide inline citations like [1], [2].\n\n"
    )
    ctx = ""
    for i, h in enumerate(hits[:4], 1):
        title = h.get("metadata", {}).get("title", f"Source {i}")
        snippet = h.get("document", "").strip()
        if len(snippet) > 1500:
            snippet = snippet[:1500] + "..."
        ctx += f"[{i}] {title}\n{snippet}\n\n"
    return f"{header}Context:\n{ctx}Question: {query}\n\nAnswer:"

def verify_equations_in_text(text: str):
    """Find LaTeX inline equations and try to parse them via SymPy."""
    eqs = re.findall(r"\$(.+?)\$", text)
    results = []
    for e in eqs:
        try:
            parsed = parse_latex(e)
            results.append({"expr": e, "status": "parsed", "notes": str(parsed)})
        except (SympifyError, Exception) as ex:
            results.append({"expr": e, "status": "failed", "notes": str(ex)})
    return results

def _extractive_fallback_answer(query: str, hits: List[Dict], top_context_n: int = 4):
    """
    Build a reasonable answer without an LLM by extracting relevant sentences
    from the top_context_n hits and composing them into a short explanation.
    """
    # gather candidate sentences from top hits
    sentences = []
    for i, h in enumerate(hits[:top_context_n], 1):
        doc = h.get("document", "") or ""
        # split into sentences
        sents = sent_tokenize(doc) if isinstance(doc, str) and doc.strip() else []
        # prefer sentences that include keywords from the query (basic heuristic)
        q_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
        picked = None
        for s in sents:
            s_lower = s.lower()
            if any(qw in s_lower for qw in q_words[:6]):
                picked = s
                break
        if not picked and sents:
            picked = sents[0]
        if picked:
            sentences.append((i, picked.strip()))
    # Compose answer: short summary + details with citations
    if not sentences:
        # last-resort: use first 200 chars of first doc
        fallback_text = hits[0]["document"][:400] if hits else "No information available."
        answer = f"Short answer (from retrieved sources): {fallback_text}"
    else:
        # Make a concise paragraph
        first_bits = " ".join([f"{s}" for _, s in sentences[:2]])
        answer = f"Short answer: {first_bits}\n\nDetails from sources:\n"
        for idx, sent in sentences:
            answer += f"- {sent} [{idx}]\n"
    return answer

def generate_answer(query, collection, top_k=5, max_length=256, temperature=0.0):
    """
    RAG pipeline:
    - retrieve top_k hits (we'll show top_k citations)
    - build prompt with top 4 contexts
    - try to generate with LLM pipeline; if not possible or result is unusable, fallback to extractive compose
    - return answer text, hits (with 'reference'), and equation checks
    """
    gen_pipeline, _ = _init_models()

    # Retrieve documents
    hits = retrieve_contexts(collection, query, top_k=top_k)

    # Ensure hits are present
    if not hits:
        return {"answer": "No relevant documents found.", "hits": [], "equation_checks": []}

    # Build LLM prompt from top 4 hits
    prompt = build_prompt(query, hits[:4])

    # Try to use LLM pipeline if available
    generated_text = ""
    if gen_pipeline is not None:
        try:
            out = gen_pipeline(
                prompt,
                max_length=max_length,
                do_sample=(temperature > 0.0),
                temperature=temperature
            )
            # pipeline output formats can vary; be defensive
            if isinstance(out, list) and len(out) > 0:
                first = out[0]
                generated_text = (
                    first.get("generated_text")
                    or first.get("summary_text")
                    or first.get("text")
                    or str(first)
                )
            else:
                generated_text = str(out)
        except Exception as e:
            # generation failed — we'll fallback below
            generated_text = ""

    # If generated text is empty or tiny, use extractive fallback
    if not generated_text or len(generated_text.strip()) < 20:
        generated_text = _extractive_fallback_answer(query, hits, top_context_n=4)

    # Build citations: include up to top_k hits as 'reference' strings for UI
    citations = []
    for i, h in enumerate(hits[:top_k], 1):
        meta = h.get("metadata", {}) or {}
        ref_str = format_ieee(meta)
        if not ref_str:
            # construct minimal reference
            title = meta.get("title") or meta.get("paper_title") or "Untitled"
            link = meta.get("link") or meta.get("url") or ""
            ref_str = f"\"{title}\"{', ' + link if link else ''}"
        citations.append({
            "reference": f"[{i}] {ref_str}",
            "metadata": meta,
            "document": h.get("document", ""),
            "distance": h.get("distance")
        })

    # Verify equations in the final answer
    eq_checks = verify_equations_in_text(generated_text)

    return {
        "answer": generated_text.strip(),
        "hits": citations,
        "equation_checks": eq_checks
    }
