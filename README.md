PhysiCalc-RAG: A Physics Research Assistant
PhysiCalc-RAG is a Retrieval-Augmented Generation (RAG) system specifically designed for the domain of physics. It allows users to ask complex, technical questions and receive accurate, evidence-based answers with proper citations from a corpus of research papers. Built with a focus on technical accuracy, the system is capable of handling specialized terminology, mathematical equations, and scientific notation, making it a reliable tool for researchers, students, and enthusiasts.

Key Features
Domain-Specific Terminology Processing: The system is trained to understand and process technical physics terms for accurate retrieval and generation.

Mathematical Equation Recognition: It extracts and validates LaTeX-formatted equations from research papers, displaying them clearly in the final answer.

Research-Backed Answers: Generates responses grounded in the retrieved document chunks, ensuring all answers are evidence-based.

Proper Citation Formatting: Provides answers with inline citations ([1], [2]), linking to the source research papers for verifiability.

Effective Chunking Strategies: Utilizes a tailored chunking strategy to preserve the context of sentences and paragraphs, which is crucial for technical documents.

Clear UX and Logical Data Flow: Features a simple and intuitive Streamlit interface that clearly guides the user from query to a well-cited answer.

Technical Accuracy Validation: Includes a basic equation verification step using SymPy to check for parsing errors.

Technical Stack & Data Flow
PhysiCalc-RAG is built on a robust RAG architecture, with a custom data ingestion pipeline and state-of-the-art open-source models.

Data Ingestion Pipeline
The system's knowledge base is built by first running a custom data ingestion pipeline to prepare the source documents.

Fetch: Connects to the arXiv API to download a configurable number of physics research papers.

Normalize: Extracts and cleans text from the downloaded PDFs, handling whitespace and formatting.

Chunk: Splits the normalized documents into smaller, meaningful chunks suitable for embedding and retrieval.

Core RAG Architecture
The RAG pipeline is built using the following components:

Embedding Model: Uses a Hugging Face SentenceTransformers model (all-MiniLM-L6-v2) to convert text chunks and user queries into vector embeddings.

Vector Database: Implements retrieval using ChromaDB, an open-source vector database that stores and indexes the document embeddings for efficient search.

Generation Model: Leverages a Hugging Face transformers model (google/flan-t5-small) to generate concise answers based on the retrieved context. A fallback to an extractive summarization method is in place if the generation model fails.

Getting Started
Follow these steps to set up and run the project locally.

Prerequisites
Python 3.8 or newer

pip

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install the required dependencies from requirements.txt:

Bash

pip install -r requirements.txt
Usage
To run the Streamlit application, execute the following command from the root directory of the project:

Bash

streamlit run app.py
On the first run, the application will automatically start the data ingestion pipeline, which may take some time as it fetches and processes papers from arXiv.

Deployed Demo
A live demo of the application is available at: [Insert Streamlit Cloud link here]

Folder Structure
.
├── .env.example              # Example file for environment variables
├── app.py                    # Main Streamlit application script
├── README.md                 # Project README file
├── requirements.txt          # Python dependencies
├── src/                      # Contains all the source code
│   ├── __init__.py           # Makes src a Python package
│   ├── demo/
│   │   ├── __init__.py
│   │   └── quick_query.py    # Standalone script for quick testing
│   ├── ingest/               # Data ingestion pipeline
│   │   ├── __init__.py
│   │   ├── fetch_arxiv.py    # Fetches papers from arXiv
│   │   ├── normalize.py      # Normalizes text and extracts equations
│   │   └── chunk.py          # Splits text into chunks
│   ├── generator/            # LLM-based answer generation logic
│   │   ├── __init__.py
│   │   └── generate.py
│   ├── index/                # Vector index creation and management
│   │   ├── __init__.py
│   │   └── build_index.py    # Builds and manages the ChromaDB index
│   ├── settings.py           # Centralized project configuration
│   └── utils_citation.py     # Utility functions for citation formatting
└── chromadb_data/            # Directory for the persistent ChromaDB files
Future Work
Integrate with other scientific literature APIs (e.g., Semantic Scholar).

Improve equation parsing and validation with more robust libraries.

Incorporate a user feedback loop to improve relevance scoring.

Enhance citation network analysis to rank the most influential papers.

Explore larger, more powerful LLMs for answer generation.
