from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore

def run_ingestion(file_path: str):
    loader =DocumentLoader()
    document =loader.load(file_path)
    
    chunker=DocumentChunker()
    chunks=chunker.chunk(document)

    embedder=DocumentEmbedder()
    embedded_chunks=embedder.embed(chunks)

    print(f"Processed {len(embedded_chunks)} chunks from {file_path}")
    
    '''print(f"\n{'='*50}")
    print(f"Pipeline complete.")
    print(f"Document: {file_path}")
    print(f"Total chunks: {len(embedded_chunks)}")
    print(f"\nFirst chunk preview:")
    print(f"Text: {embedded_chunks[0]['text'][:200]}")
    print(f"Source: {embedded_chunks[0]['source']}")
    print(f"Type: {embedded_chunks[0]['type']}")
    print(f"Chunk ID: {embedded_chunks[0]['chunk_id']}")
    print(f"Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")
    print(f"{'='*50}")'''

     # Step 4 - Store in vector DB
    vector_store = VectorStore()
    stored = vector_store.upsert_chunks(embedded_chunks)

    # Step 5 - Verify
    stats = vector_store.get_index_stats()

    print(f"\n{'='*50}")
    print(f"Pipeline complete.")
    print(f"Document: {file_path}")
    print(f"Chunks stored: {stored}")
    print(f"Total vectors in DB: {stats['total_vectors']}")
    print(f"Index dimensions: {stats['dimensions']}")
    print(f"{'='*50}")

    
    return embedded_chunks

if __name__ == "__main__":
    run_ingestion(r"C:\Users\ayushi.awasthi\Documents\Books\Ghosal-SOCIOPOLITICALDIMENSIONSRAPE-2009.pdf")