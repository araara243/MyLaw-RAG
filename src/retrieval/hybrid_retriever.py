"""
Hybrid Retriever for Malaysian Legal RAG

This module implements a hybrid search combining:
1. Semantic Search: Vector similarity using ChromaDB embeddings
2. Keyword Search: BM25-based exact term matching

Hybrid search is critical for legal documents because:
- Legal terms like "consideration" have specific meanings
- Semantic search alone may miss exact terminology
- BM25 provides precision; vectors provide recall

The retriever uses Reciprocal Rank Fusion (RRF) to combine results.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

from rank_bm25 import BM25Okapi

from config import (
    RAGConfig,
    get_vector_db_dir,
    setup_logging
)

# Configure logging
logger = setup_logging(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    chunk_id: str
    content: str
    act_name: str
    act_number: int
    section_number: str
    section_title: str
    score: float
    retrieval_method: str  # "semantic", "keyword", or "hybrid"


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.
    
    Uses ChromaDB for semantic search and BM25 for keyword search,
    with Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            config: Optional RAGConfig object. If None, uses defaults.
        """
        self.config = config or RAGConfig()
        
        self.collection_name = self.config.collection_name
        self.semantic_weight = self.config.semantic_weight
        self.keyword_weight = self.config.keyword_weight
        self.rrf_k = self.config.rrf_k
        
        # Initialize components
        self._collection: Any = None
        self._bm25: Optional[BM25Okapi] = None
        self._documents: List[str] = []
        self._doc_ids: List[str] = []
        self._doc_metadata: List[Dict[str, Any]] = []
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize ChromaDB and BM25 index."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Load ChromaDB collection
            db_path = str(get_vector_db_dir())
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self._collection = client.get_collection(name=self.collection_name)
            
            # Get all documents for BM25 indexing
            all_docs = self._collection.get(include=["documents", "metadatas"])
            
            if not all_docs or not all_docs["ids"]:
                logger.warning(f"Collection {self.collection_name} is empty or not found.")
                return

            self._doc_ids = all_docs["ids"]
            # Ensure documents are strings, handle potential None values
            self._documents = [doc if doc is not None else "" for doc in all_docs["documents"]]
            self._doc_metadata = all_docs["metadatas"]
            
            # Build BM25 index
            tokenized_docs = [self._tokenize(doc) for doc in self._documents]
            self._bm25 = BM25Okapi(tokenized_docs)
            
            logger.info(
                f"Initialized HybridRetriever with {len(self._documents)} documents"
            )
        except Exception as e:
            logger.error(f"Failed to initialize HybridRetriever: {e}")
            raise
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Uses simple whitespace + punctuation tokenization.
        Preserves legal terms like "Section 10" as single tokens.
        """
        if not text:
            return []
            
        # Lowercase
        text = text.lower()
        
        # Keep "section X" together
        text = re.sub(r"section\s+(\d+[a-z]*)", r"section_\1", text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text)
        
        return tokens
    
    def _semantic_search(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Perform semantic search using ChromaDB.
        
        Returns list of (doc_id, distance) tuples.
        """
        if not self._collection:
            logger.error("Collection not initialized.")
            return []
            
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["distances"]
            )
            
            if not results["ids"]:
                return []

            # Convert distances to similarity scores (1 - distance for cosine)
            return [
                (doc_id, 1 - distance)
                for doc_id, distance in zip(
                    results["ids"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _keyword_search(
        self,
        query: str,
        n_results: int
    ) -> List[Tuple[str, float]]:
        """
        Perform keyword search using BM25.
        
        Returns list of (doc_id, score) tuples.
        """
        if not self._bm25:
            logger.warning("BM25 index not initialized.")
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top N indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:n_results]
        
        # Return (doc_id, score) pairs
        return [
            (self._doc_ids[i], scores[i])
            for i in top_indices
            if scores[i] > 0  # Filter zero scores
        ]
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each method
        
        Args:
            semantic_results: List of (doc_id, score) from semantic search.
            keyword_results: List of (doc_id, score) from keyword search.
        
        Returns:
            Dictionary mapping doc_id to combined RRF score.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        
        # Add semantic ranks
        for rank, (doc_id, _) in enumerate(semantic_results, start=1):
            rrf_scores[doc_id] += self.semantic_weight / (self.rrf_k + rank)
        
        # Add keyword ranks
        for rank, (doc_id, _) in enumerate(keyword_results, start=1):
            rrf_scores[doc_id] += self.keyword_weight / (self.rrf_k + rank)
        
        return dict(rrf_scores)
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        method: str = "hybrid"
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant legal chunks for a query.
        
        Args:
            query: The user's legal question.
            n_results: Number of results to return.
            method: "hybrid", "semantic", or "keyword".
        
        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        try:
            # Perform searches based on method
            semantic_results: List[Tuple[str, float]] = []
            keyword_results: List[Tuple[str, float]] = []
            
            if method in ("hybrid", "semantic"):
                semantic_results = self._semantic_search(query, n_results * 2)
            
            if method in ("hybrid", "keyword"):
                keyword_results = self._keyword_search(query, n_results * 2)
            
            # Combine results
            combined_scores: Dict[str, float] = {}
            if method == "hybrid":
                combined_scores = self._reciprocal_rank_fusion(
                    semantic_results, keyword_results
                )
            elif method == "semantic":
                combined_scores = {doc_id: score for doc_id, score in semantic_results}
            else:  # keyword
                combined_scores = {doc_id: score for doc_id, score in keyword_results}
            
            # Sort by score and take top N
            sorted_ids = sorted(
                combined_scores.keys(),
                key=lambda x: combined_scores[x],
                reverse=True
            )[:n_results]
            
            # Build result objects
            results = []
            for doc_id in sorted_ids:
                if doc_id not in self._doc_ids:
                    continue 

                # Find document index
                idx = self._doc_ids.index(doc_id)
                metadata = self._doc_metadata[idx]
                
                result = RetrievalResult(
                    chunk_id=doc_id,
                    content=self._documents[idx],
                    act_name=metadata.get("act_name", ""),
                    act_number=metadata.get("act_number", 0),
                    section_number=metadata.get("section_number", ""),
                    section_title=metadata.get("section_title", ""),
                    score=combined_scores[doc_id],
                    retrieval_method=method
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {e}")
            return []
    
    def format_context(
        self,
        results: List[RetrievalResult],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieval results as context for the LLM.
        
        Args:
            results: List of RetrievalResult objects.
            include_metadata: Whether to include citation metadata.
        
        Returns:
            Formatted context string.
        """
        context_parts = []
        
        for i, result in enumerate(results, start=1):
            if include_metadata:
                header = (
                    f"[Source {i}: {result.act_name}, "
                    f"Section {result.section_number}]"
                )
                if result.section_title:
                    header = header[:-1] + f" - {result.section_title}]"
            else:
                header = f"[Source {i}]"
            
            context_parts.append(f"{header}\n{result.content}")
        
        return "\n\n---\n\n".join(context_parts)


def test_retriever():
    """Test the hybrid retriever with sample queries."""
    try:
        retriever = HybridRetriever()
        
        test_queries = [
            "What is the definition of consideration in contract law?",
            "Section 10 free consent",
            "housing developer license requirements",
            "specific performance contract enforcement",
        ]
        
        print("=" * 70)
        print("Hybrid Retriever Test")
        print("=" * 70)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            # Test all methods
            for method in ["semantic", "keyword", "hybrid"]:
                results = retriever.retrieve(query, n_results=3, method=method)
                print(f"\n  [{method.upper()}]")
                for r in results:
                    print(
                        f"    - {r.act_name} Section {r.section_number} "
                        f"(score: {r.score:.4f})"
                    )
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_retriever()
