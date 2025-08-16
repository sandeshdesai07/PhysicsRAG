# 🌌 PhysicsRAG
**A Physics Research Assistant**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-orange) ![arXiv](https://img.shields.io/badge/arXiv-API-red) ![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green) ![License](https://img.shields.io/badge/License-MIT-green)

**PhysiCalc-RAG** is a **Retrieval-Augmented Generation (RAG)** system that answers **complex physics queries** with **research-backed responses**, including **equations**, **scientific notation**, and **inline citations**. Ideal for **researchers, students, and enthusiasts**.

---

## 🎨 Features

- 🔹 **Domain-Specific Physics Terminology**
- 🔹 **Mathematical Equation Recognition** (LaTeX + SymPy validation)
- 🔹 **Research-Backed Answers** with inline citations ([1], [2])
- 🔹 **Context-Preserving Chunking** for technical documents
- 🔹 **Intuitive Streamlit UI** for seamless user experience
- 🔹 **Technical Accuracy Validation**

---

## 🎬 Live Demo

[[🔗 Streamlit Cloud Demo](https://physicsrag-yud3dkfvgmkcpyz4bnoqrc.streamlit.app/)]

![PhysiCalc-RAG Demo](assets/demo.gif)

---

## ⚡ Quick Start (3 Steps)

| Step | Action | Command |
|------|--------|---------|
| 1️⃣ | Clone Repo | `git clone https://github.com/gh repo clone sandeshdesai07/PhysicsRAG.git` |
| 2️⃣ | Install Dependencies | `pip install -r requirements.txt` |
| 3️⃣ | Run App | `streamlit run app.py` |

> ⚠️ **Note:** First run triggers data ingestion from arXiv (~few minutes).

---

## 🗂️ Folder Structure

```text
PhysicsRAG/
├── .env.example            # Example environment variables
├── app.py                  # Main Streamlit app
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── demo/
│   │   ├── __init__.py
│   │   └── quick_query.py  # Standalone test script
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── fetch_arxiv.py
│   │   ├── normalize.py
│   │   └── chunk.py
│   ├── generator/
│   │   ├── __init__.py
│   │   └── generate.py
│   ├── index/
│   │   ├── __init__.py
│   │   └── build_index.py
│   ├── settings.py
│   └── utils_citation.py
└── chromadb_data/          # Persistent ChromaDB storage

| Component           | Technology                                                  |
| ------------------- | ----------------------------------------------------------- |
| Embeddings          | Hugging Face `SentenceTransformers` (all-MiniLM-L6-v2)      |
| Vector Database     | ChromaDB                                                    |
| Generation Model    | `google/flan-t5-small` (fallback: extractive summarization) |
| UI                  | Streamlit                                                   |
| Equation Validation | SymPy                                                       |

