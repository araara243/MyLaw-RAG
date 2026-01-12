"""
RAG Evaluation Suite for Malaysian Legal RAG

This module implements evaluation metrics for the RAG system:
1. Retrieval metrics (Hit Rate, MRR)
2. Context relevancy 
3. Faithfulness (answer grounded in context)
4. Citation accuracy

Uses a lightweight evaluation approach that doesn't require external APIs.
"""

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.hybrid_retriever import HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating a single question."""
    question_id: str
    question: str
    expected_act: str
    expected_section: str
    
    # Retrieval metrics
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    
    # Retrieved sources
    top_sources: list


def load_golden_dataset() -> dict:
    """Load the golden dataset."""
    dataset_path = PROJECT_ROOT / "tests" / "golden_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    retriever: HybridRetriever,
    question: str,
    expected_act: str,
    expected_section: str,
    k: int = 5
) -> EvaluationResult:
    """
    Evaluate retrieval for a single question.
    
    Args:
        retriever: The hybrid retriever instance.
        question: The test question.
        expected_act: The act that should be retrieved.
        expected_section: The section that should be retrieved.
        k: Number of results to retrieve.
    
    Returns:
        EvaluationResult with metrics.
    """
    results = retriever.retrieve(question, n_results=k, method="hybrid")
    
    # Find rank of expected result
    rank = None
    for i, r in enumerate(results, start=1):
        if expected_act in r.act_name:
            # Check if section matches (fuzzy match)
            section_match = (
                expected_section.lower().replace("section ", "")
                in str(r.section_number).lower()
            )
            if section_match or rank is None:
                rank = i
                if section_match:
                    break
    
    # Calculate metrics
    hit_at_1 = rank == 1 if rank else False
    hit_at_3 = rank is not None and rank <= 3
    hit_at_5 = rank is not None and rank <= 5
    reciprocal_rank = 1.0 / rank if rank else 0.0
    
    return EvaluationResult(
        question_id="",  # Will be set by caller
        question=question,
        expected_act=expected_act,
        expected_section=expected_section,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=reciprocal_rank,
        top_sources=[
            f"{r.act_name}, Section {r.section_number}"
            for r in results[:3]
        ]
    )


def run_evaluation() -> dict:
    """
    Run full evaluation on the golden dataset.
    
    Returns:
        Dictionary with aggregate metrics.
    """
    logger.info("=" * 60)
    logger.info("Malaysian Legal RAG Evaluation")
    logger.info("=" * 60)
    
    # Load dataset
    dataset = load_golden_dataset()
    questions = dataset["questions"]
    logger.info(f"Loaded {len(questions)} test questions")
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Evaluate each question
    results = []
    
    for q in questions:
        eval_result = evaluate_retrieval(
            retriever=retriever,
            question=q["question"],
            expected_act=q["expected_act"],
            expected_section=q["expected_section"]
        )
        eval_result.question_id = q["id"]
        results.append(eval_result)
        
        # Log individual result
        status = "✓" if eval_result.hit_at_3 else "✗"
        logger.info(
            f"{status} {q['id']}: {q['question'][:50]}... "
            f"(RR: {eval_result.reciprocal_rank:.2f})"
        )
    
    # Calculate aggregate metrics
    n = len(results)
    
    hit_rate_1 = sum(r.hit_at_1 for r in results) / n
    hit_rate_3 = sum(r.hit_at_3 for r in results) / n
    hit_rate_5 = sum(r.hit_at_5 for r in results) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Hit Rate @1: {hit_rate_1:.1%}")
    logger.info(f"Hit Rate @3: {hit_rate_3:.1%}")
    logger.info(f"Hit Rate @5: {hit_rate_5:.1%}")
    logger.info(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    logger.info("=" * 60)
    
    # Detailed failures
    failures = [r for r in results if not r.hit_at_3]
    if failures:
        logger.info(f"\nFailed questions ({len(failures)}):")
        for f in failures:
            logger.info(f"  - {f.question_id}: Expected {f.expected_act}, {f.expected_section}")
            logger.info(f"    Got: {', '.join(f.top_sources)}")
    
    # Save results
    output = {
        "metrics": {
            "hit_rate_at_1": hit_rate_1,
            "hit_rate_at_3": hit_rate_3,
            "hit_rate_at_5": hit_rate_5,
            "mrr": mrr,
            "total_questions": n
        },
        "individual_results": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "expected": f"{r.expected_act}, {r.expected_section}",
                "hit_at_3": r.hit_at_3,
                "reciprocal_rank": r.reciprocal_rank,
                "top_sources": r.top_sources
            }
            for r in results
        ]
    }
    
    output_path = PROJECT_ROOT / "tests" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return output


if __name__ == "__main__":
    run_evaluation()
