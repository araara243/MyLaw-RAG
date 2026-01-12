"""
Vector Database Ingestion for Malaysian Legal RAG

This module handles:
- Embedding legal chunks using sentence-transformers (local, free)
- Storing vectors in ChromaDB for local retrieval
- Metadata management for citation

ChromaDB is used for MVP as it's local and requires no external dependencies.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_processed_data_dir() -> Path:
    """Get the processed data directory."""
    return get_project_root() / "data" / "processed"


def get_vector_db_dir() -> Path:
    """Get the vector database directory, creating it if necessary."""
    db_dir = get_project_root() / "data" / "vector_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def load_all_chunks() -> list[dict]:
    """
    Load all chunk files from the processed directory.
    
    Returns:
        List of all chunks across all documents.
    """
    processed_dir = get_processed_data_dir()
    chunk_files = list(processed_dir.glob("*_chunks.json"))
    
    all_chunks = []
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
    
    logger.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files")
    return all_chunks


def create_chroma_collection(
    collection_name: str = "malaysian_legal_acts"
) -> "chromadb.Collection":
    """
    Create or get a ChromaDB collection for legal documents.
    
    Args:
        collection_name: Name of the collection.
    
    Returns:
        ChromaDB Collection object.
    """
    import chromadb
    from chromadb.config import Settings
    
    # Initialize ChromaDB with persistent storage
    db_path = str(get_vector_db_dir())
    
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection
    # Using default embedding function (requires sentence-transformers)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Cosine similarity
    )
    
    logger.info(f"ChromaDB collection '{collection_name}' ready at {db_path}")
    return collection


def ingest_chunks_to_chroma(
    chunks: list[dict],
    collection: "chromadb.Collection",
    batch_size: int = 50
) -> int:
    """
    Ingest legal chunks into ChromaDB.
    
    Args:
        chunks: List of chunk dictionaries.
        collection: ChromaDB collection.
        batch_size: Number of chunks to insert per batch.
    
    Returns:
        Number of chunks ingested.
    """
    # Check existing documents
    existing_ids = set(collection.get()["ids"])
    
    # Filter out already ingested chunks
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    
    if not new_chunks:
        logger.info("All chunks already ingested")
        return 0
    
    logger.info(f"Ingesting {len(new_chunks)} new chunks (skipping {len(chunks) - len(new_chunks)} existing)")
    
    # Prepare data for insertion
    ids = []
    documents = []
    metadatas = []
    
    for chunk in new_chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["content"])
        metadatas.append({
            "act_name": chunk["act_name"],
            "act_number": chunk["act_number"],
            "part": chunk.get("part") or "",
            "section_number": chunk.get("section_number") or "",
            "section_title": chunk.get("section_title") or "",
            "token_count": chunk["token_count"],
        })
    
    # Insert in batches
    total_inserted = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
        
        total_inserted += len(batch_ids)
        logger.info(f"Inserted batch {i // batch_size + 1}: {len(batch_ids)} chunks")
    
    return total_inserted


def test_retrieval(
    collection: "chromadb.Collection",
    query: str,
    n_results: int = 3
) -> list[dict]:
    """
    Test retrieval from the collection.
    
    Args:
        collection: ChromaDB collection.
        query: Test query string.
        n_results: Number of results to return.
    
    Returns:
        List of retrieved documents with metadata.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i][:200] + "...",  # Truncate
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return retrieved


def run_ingestion() -> dict:
    """
    Run the full ingestion pipeline.
    
    Returns:
        Dictionary with ingestion statistics.
    """
    logger.info("=" * 60)
    logger.info("Starting Vector Database Ingestion")
    logger.info("=" * 60)
    
    # Load chunks
    chunks = load_all_chunks()
    
    if not chunks:
        logger.error("No chunks found to ingest")
        return {"error": "No chunks found"}
    
    # Create collection
    collection = create_chroma_collection()
    
    # Ingest chunks
    ingested = ingest_chunks_to_chroma(chunks, collection)
    
    # Get collection stats
    count = collection.count()
    
    # Test retrieval
    logger.info("\n" + "-" * 40)
    logger.info("Testing retrieval...")
    test_query = "What is the definition of consideration in contract law?"
    results = test_retrieval(collection, test_query)
    
    logger.info(f"\nTest query: '{test_query}'")
    for i, result in enumerate(results, 1):
        logger.info(f"\nResult {i}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Act: {result['metadata']['act_name']}")
        logger.info(f"  Section: {result['metadata']['section_number']}")
        logger.info(f"  Distance: {result['distance']:.4f}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ingestion Summary:")
    logger.info(f"  Chunks ingested: {ingested}")
    logger.info(f"  Total in collection: {count}")
    logger.info(f"  Vector DB path: {get_vector_db_dir()}")
    logger.info("=" * 60)
    
    return {
        "chunks_ingested": ingested,
        "total_in_collection": count,
        "db_path": str(get_vector_db_dir())
    }


if __name__ == "__main__":
    run_ingestion()
