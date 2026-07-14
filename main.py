import os
import logging
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)

def run_ingestion(file_path: str):
    # One file only — no loops here
    loader = DocumentLoader()
    document = loader.load(file_path)

    if not document["text"].strip():
        print(f"Skipping empty document: {file_path}")
        return None

    chunker = DocumentChunker()
    chunks = chunker.chunk(document)

    embedder = DocumentEmbedder()
    embedded_chunks = embedder.embed(chunks)

    vector_store = VectorStore()
    stored = vector_store.upsert_chunks(embedded_chunks)

    stats = vector_store.get_index_stats()

    print(f"\n{'='*50}")
    print(f"Document: {os.path.basename(file_path)}")
    print(f"Chunks stored: {stored}")
    print(f"Total vectors in DB: {stats['total_vectors']}")
    print(f"{'='*50}")

    return embedded_chunks


if __name__ == "__main__":
    docs_folder = r"C:\Users\ayushi.awasthi\Documents\Python\rag-pipeline\data\documents"

    # Filter both ~$ temp files AND non-PDFs
    pdf_files = [
        os.path.join(docs_folder, f)
        for f in os.listdir(docs_folder)
        if f.endswith(".pdf") and not f.startswith("~$")
    ]

    print(f"Found {len(pdf_files)} documents to process")

    for file_path in pdf_files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        try:
            run_ingestion(file_path)
        except Exception as e:
            logger.error(f"Failed: {os.path.basename(file_path)}: {e}")
            print(f"Skipping: {os.path.basename(file_path)}")
            continue

    print("\nAll documents processed successfully.")