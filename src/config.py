"""
Configuration management for MyLaw-RAG.

This module centralizes all configuration settings, path management,
and logging setup to ensure consistency across the application.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration settings for the RAG pipeline."""
    
    # Retrieval Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    
    # Hybrid Search Weights
    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    rrf_k: int = 60
    
    # Vector DB
    collection_name: str = "malaysian_legal_acts"
    
    # Model Settings (defaults)
    embedding_model: str = "all-MiniLM-L6-v2"  # implicitly used by sentence-transformers
    llm_model: str = "gemini-2.0-flash-lite"
    temperature: float = 0.1


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_processed_dir() -> Path:
    """Get the processed data directory."""
    return get_data_dir() / "processed"


def get_vector_db_dir() -> Path:
    """Get the vector database directory."""
    return get_data_dir() / "vector_db"


def setup_logging(name: str) -> logging.Logger:
    """
    Setup a standard logger with consistent formatting.
    
    Args:
        name: The name of the logger (usually __name__).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist to avoid duplicates
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger
