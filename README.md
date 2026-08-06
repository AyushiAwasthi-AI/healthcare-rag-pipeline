# healthcare-rag-pipeline
Production RAG pipeline for healthcare documents with HIPAA compliance Visibility: Public

# Healthcare RAG Pipeline 🏥

Production-grade Retrieval-Augmented Generation pipeline 
for healthcare documents with HIPAA compliance.

## Architecture

Ingestion Pipeline:
Document → File Type Check → OCR → Classifier → 
Chunker → Embedder → Vector DB (Pinecone)

Query Pipeline (WIP):
Query → Query Processor → Hybrid Search → 
Re-ranker → Prompt → LLM → RAGAS Evaluation

## Tech Stack

- **Vector DB:** Pinecone
- **Embeddings:** Sentence Transformers (ClinicalBERT)
- **LLM:** OpenAI GPT-4
- **Evaluation:** RAGAS
- **OCR:** Tesseract + AWS Textract
- **Framework:** LangChain
- **API:** FastAPI (coming)

## Key Features

- HIPAA-compliant chunk deletion by document source
- Dynamic chunk sizing based on document length
- Hybrid search: semantic + BM25 keyword
- Cross-encoder re-ranking
- Domain-specific medical embeddings
- OCR support for scanned healthcare documents
- RAGAS automated evaluation

## Project Structure

ingestion/
  loader.py      - Document loading + OCR
  chunker.py     - 4 chunking strategies
  embedder.py    - Domain-specific embeddings
  vector_store.py - Pinecone operations

query/           - Coming
generation/      - Coming
evaluation/      - Coming
api/             - FastAPI layer (coming)

## Setup

1. Clone repo
2. Create virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate.bat
4. Install: pip install -r requirements.txt
5. Copy .env.example to .env and add your API keys
6. Run: python main.py
