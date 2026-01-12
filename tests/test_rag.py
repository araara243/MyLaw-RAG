"""
Unit Tests for Malaysian Legal RAG

Tests for:
- AGC Scraper functionality
- Text extraction and cleaning  
- Semantic chunking
- Hybrid retrieval accuracy
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestAGCScraper:
    """Tests for the AGC scraper module."""
    
    def test_construct_pdf_url_english(self):
        """Test English PDF URL construction."""
        from ingestion.agc_scraper import construct_pdf_url
        
        url = construct_pdf_url(136, "EN")
        assert "Act%20136.pdf" in url
        assert "LOM/EN" in url
    
    def test_construct_pdf_url_malay(self):
        """Test Malay PDF URL construction."""
        from ingestion.agc_scraper import construct_pdf_url
        
        url = construct_pdf_url(136, "BM")
        assert "Akta%20136.pdf" in url
        assert "LOM/BM" in url
    
    def test_mvp_acts_defined(self):
        """Test that MVP acts are properly defined."""
        from ingestion.agc_scraper import MVP_ACTS
        
        assert len(MVP_ACTS) >= 3
        act_numbers = [act["act_no"] for act in MVP_ACTS]
        assert 136 in act_numbers  # Contracts Act
        assert 137 in act_numbers  # Specific Relief Act
        assert 118 in act_numbers  # Housing Development Act


class TestTextExtractor:
    """Tests for the text extraction module."""
    
    def test_clean_legal_text_removes_headers(self):
        """Test that AGC headers are removed."""
        from ingestion.text_extractor import clean_legal_text
        
        text = "AGC Malaysia\nSection 1. This is the law.\nPage 1 of 10"
        cleaned = clean_legal_text(text)
        
        assert "AGC Malaysia" not in cleaned
        assert "Page 1 of 10" not in cleaned
        assert "Section 1" in cleaned
    
    def test_clean_legal_text_normalizes_whitespace(self):
        """Test whitespace normalization."""
        from ingestion.text_extractor import clean_legal_text
        
        text = "Section 1.   Too many   spaces here.\n\n\n\nToo many newlines."
        cleaned = clean_legal_text(text)
        
        assert "   " not in cleaned  # Multiple spaces removed
        assert "\n\n\n" not in cleaned  # Multiple newlines normalized


class TestChunker:
    """Tests for the semantic chunking module."""
    
    def test_find_sections(self):
        """Test section detection."""
        from ingestion.chunker import find_sections
        
        # Note: Text format matches real PDF extraction (section headers at line start)
        text = """Section 1. Short title
This Act may be cited as the Test Act.

Section 2. Interpretation
In this Act, unless the context otherwise requires—
"""
        
        sections = find_sections(text)
        assert len(sections) == 2
        assert sections[0]["section_number"] == "1"
        assert sections[1]["section_number"] == "2"
    
    def test_find_sections_with_subsections(self):
        """Test that subsection patterns like 2A are detected."""
        from ingestion.chunker import find_sections
        
        # Note: Text format matches real PDF extraction (section headers at line start)
        text = """Section 5A. Additional provisions
This section was added later.
"""
        
        sections = find_sections(text)
        assert len(sections) == 1
        assert sections[0]["section_number"] == "5A"


class TestHybridRetriever:
    """Tests for the hybrid retriever."""
    
    @pytest.fixture
    def retriever(self):
        """Create retriever instance."""
        from retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever()
    
    def test_retriever_initialization(self, retriever):
        """Test that retriever initializes with documents."""
        assert retriever._documents is not None
        assert len(retriever._documents) > 0
    
    def test_semantic_search_returns_results(self, retriever):
        """Test semantic search returns results."""
        results = retriever._semantic_search(
            "What is consideration in contract law?",
            n_results=3
        )
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
    
    def test_keyword_search_returns_results(self, retriever):
        """Test keyword search returns results."""
        results = retriever._keyword_search(
            "Section 10 free consent",
            n_results=3
        )
        assert len(results) > 0
    
    def test_hybrid_search_combines_methods(self, retriever):
        """Test hybrid search returns combined results."""
        results = retriever.retrieve(
            "housing developer license",
            n_results=5,
            method="hybrid"
        )
        
        assert len(results) > 0
        assert all(r.retrieval_method == "hybrid" for r in results)
    
    def test_contracts_act_retrieval(self, retriever):
        """Test that Contracts Act queries return Contracts Act sections."""
        results = retriever.retrieve(
            "What is the definition of consideration?",
            n_results=3,
            method="hybrid"
        )
        
        # At least one result should be from Contracts Act
        contracts_results = [r for r in results if "Contracts" in r.act_name]
        assert len(contracts_results) > 0
    
    def test_housing_act_retrieval(self, retriever):
        """Test that housing queries return Housing Development Act sections."""
        results = retriever.retrieve(
            "housing developer license requirements",
            n_results=3,
            method="hybrid"
        )
        
        # At least one result should be from Housing Development Act
        housing_results = [r for r in results if "Housing" in r.act_name]
        assert len(housing_results) > 0
    
    def test_specific_relief_retrieval(self, retriever):
        """Test that specific performance queries return Specific Relief Act."""
        results = retriever.retrieve(
            "specific performance of contract",
            n_results=3,
            method="hybrid"
        )
        
        # At least one result should be from Specific Relief Act
        relief_results = [r for r in results if "Specific Relief" in r.act_name]
        assert len(relief_results) > 0


class TestGoldenDataset:
    """Tests using the golden dataset."""
    
    @pytest.fixture
    def golden_dataset(self):
        """Load golden dataset."""
        dataset_path = PROJECT_ROOT / "tests" / "golden_dataset.json"
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    @pytest.fixture
    def retriever(self):
        """Create retriever instance."""
        from retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever()
    
    def test_golden_dataset_structure(self, golden_dataset):
        """Test golden dataset has expected structure."""
        assert "questions" in golden_dataset
        assert len(golden_dataset["questions"]) == 20
        
        for q in golden_dataset["questions"]:
            assert "question" in q
            assert "expected_act" in q
            assert "ground_truth" in q
    
    def test_retrieval_accuracy_on_golden_dataset(self, golden_dataset, retriever):
        """Test retrieval accuracy on golden dataset (at least 70% should match expected act)."""
        correct = 0
        total = len(golden_dataset["questions"])
        
        for q in golden_dataset["questions"]:
            results = retriever.retrieve(q["question"], n_results=3, method="hybrid")
            
            # Check if expected act appears in top 3 results
            for r in results:
                if q["expected_act"] in r.act_name:
                    correct += 1
                    break
        
        accuracy = correct / total
        print(f"\nRetrieval Accuracy: {correct}/{total} = {accuracy:.1%}")
        
        # We expect at least 70% accuracy
        assert accuracy >= 0.7, f"Retrieval accuracy {accuracy:.1%} is below 70% threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
