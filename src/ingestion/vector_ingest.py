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
from typing import Optional, List, Dict, Any, Union

from config import (
    RAGConfig,
    get_processed_dir,
    get_vector_db_dir,
    setup_logging
)

# Configure logging
logger = setup_logging(__name__)


def load_all_chunks() -> List[Dict[str, Any]]:
    """
    Load all chunk files from the processed directory.
    
    Returns:
        List of all chunks across all documents.
    """
    processed_dir = get_processed_dir()
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return []

    chunk_files = list(processed_dir.glob("*_chunks.json"))
    
    all_chunks = []
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to load chunks from {chunk_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files")
    return all_chunks


def create_chroma_collection(
    collection_name: str
) -> Any:
    """
    Create or get a ChromaDB collection for legal documents.
    
    Args:
        collection_name: Name of the collection.
    
    Returns:
        ChromaDB Collection object.
    
    Raises:
        ImportError: If chromadb is not installed.
        Exception: If database initialization fails.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        logger.error("ChromaDB not installed. Please install it.")
        raise

    try:
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
    except Exception as e:
        logger.error(f"Failed to create/get ChromaDB collection: {e}")
        raise


def ingest_chunks_to_chroma(
    chunks: List[Dict[str, Any]],
    collection: Any,
    batch_size: int = 50
) -> int:
    """
    Ingest legal chunks into ChromaDB.
    Uses upsert to update existing chunks with new content.
    
    Args:
        chunks: List of chunk dictionaries.
        collection: ChromaDB collection.
        batch_size: Number of chunks to insert per batch.
    
    Returns:
        Number of chunks ingested.
    """
    try:
        logger.info(f"Ingesting/Updating {len(chunks)} chunks into ChromaDB")
        
        # Prepare data for insertion
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            metadatas.append({
                "act_name": chunk["act_name"],
                "act_number": chunk["act_number"],
                "part": chunk.get("part") or "",
                "section_number": chunk.get("section_number") or "",
                "section_title": chunk.get("section_title") or "",
                # Ensure values are strings or numbers, no Nones
                "token_count": chunk["token_count"],
            })
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, len(ids), batch_size):
            try:
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_meta
                )
                
                total_inserted += len(batch_ids)
                logger.info(f"Processed batch {i // batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as batch_error:
                logger.error(f"Error processing batch {i // batch_size + 1}: {batch_error}")
                continue
        
        return total_inserted
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 0


def test_retrieval(
    collection: Any,
    query: str,
    n_results: int = 3
) -> List[Dict[str, Any]]:
    """
    Test retrieval from the collection.
    
    Args:
        collection: ChromaDB collection.
        query: Test query string.
        n_results: Number of results to return.
    
    Returns:
        List of retrieved documents with metadata.
    """
    try:
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
    except Exception as e:
        logger.error(f"Test retrieval failed: {e}")
        return []


def run_ingestion() -> Dict[str, Union[int, str]]:
    """
    Run the full ingestion pipeline.
    
    Returns:
        Dictionary with ingestion statistics.
    """
    config = RAGConfig()
    
    logger.info("=" * 60)
    logger.info("Starting Vector Database Ingestion")
    logger.info("=" * 60)
    
    # Load chunks
    chunks = load_all_chunks()
    
    if not chunks:
        logger.error("No chunks found to ingest")
        return {"error": "No chunks found"}
    
    try:
        # Create collection
        collection = create_chroma_collection(config.collection_name)
        
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
    except Exception as e:
        logger.error(f"Ingestion process failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    run_ingestion()
