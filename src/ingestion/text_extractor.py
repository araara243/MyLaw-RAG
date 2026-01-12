"""
PDF Text Extraction and Cleaning

This module extracts text from Malaysian legal PDFs and cleans it
for downstream RAG processing.

Key cleaning operations:
- Remove headers, footers, page numbers
- Remove "AGC Malaysia" stamps and watermarks
- Clean whitespace and normalize unicode
- Preserve section structure for semantic chunking
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_raw_data_dir() -> Path:
    """Get the raw data directory."""
    return get_project_root() / "data" / "raw"


def get_processed_data_dir() -> Path:
    """Get the processed data directory, creating it if necessary."""
    processed_dir = get_project_root() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        Extracted text as a string.
    """
    logger.info(f"Extracting text from: {pdf_path.name}")
    
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        
        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
        return full_text
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def clean_legal_text(text: str) -> str:
    """
    Clean extracted legal text by removing noise.
    
    This function:
    - Removes common header/footer patterns
    - Removes page numbers
    - Removes AGC stamps
    - Normalizes whitespace
    - Preserves section structure
    
    Args:
        text: Raw extracted text.
    
    Returns:
        Cleaned text.
    """
    if not text:
        return ""
    
    # Remove common AGC header patterns
    patterns_to_remove = [
        # AGC headers/stamps
        r"AGC\s*Malaysia",
        r"Attorney\s*General['']?s?\s*Chambers",
        r"Jabatan\s*Peguam\s*Negara",
        
        # Page markers
        r"Page\s*\d+\s*of\s*\d+",
        r"Mukasurat\s*\d+\s*daripada\s*\d+",
        
        # Common footer patterns
        r"www\.agc\.gov\.my",
        r"http[s]?://\S+",
        
        # Reprint markers (but keep the year info)
        r"Incorporating\s*all\s*amendments\s*up\s*to\s*\d+\s*\w+\s*\d{4}",
        
        # Loose page numbers at line start/end
        r"^\s*\d{1,3}\s*$",
    ]
    
    cleaned = text
    
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Normalize multiple newlines (keep max 2)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    # Normalize multiple spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    # Clean up lines that are just whitespace
    cleaned = "\n".join(
        line for line in cleaned.split("\n") 
        if line.strip()
    )
    
    return cleaned.strip()


def extract_act_metadata(text: str, filename: str) -> dict:
    """
    Extract metadata from the Act text and filename.
    
    Args:
        text: The cleaned Act text.
        filename: The PDF filename.
    
    Returns:
        Dictionary containing act metadata.
    """
    metadata = {
        "filename": filename,
        "act_number": None,
        "act_name": None,
        "language": None,
    }
    
    # Extract from filename: Act_136_Contracts Act 1950_EN.pdf
    filename_match = re.match(
        r"Act_(\d+)_(.+)_(EN|BM)\.pdf",
        filename,
        re.IGNORECASE
    )
    
    if filename_match:
        metadata["act_number"] = int(filename_match.group(1))
        metadata["act_name"] = filename_match.group(2)
        metadata["language"] = filename_match.group(3).upper()
    
    # Try to extract from text if not found
    if not metadata["act_number"]:
        act_num_match = re.search(r"Act\s*(\d+)", text, re.IGNORECASE)
        if act_num_match:
            metadata["act_number"] = int(act_num_match.group(1))
    
    return metadata


def process_pdf(pdf_path: Path) -> Optional[dict]:
    """
    Process a single PDF file: extract, clean, and save.
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        Dictionary with processed data, or None on error.
    """
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        logger.error(f"No text extracted from {pdf_path.name}")
        return None
    
    # Clean text
    cleaned_text = clean_legal_text(raw_text)
    
    # Extract metadata
    metadata = extract_act_metadata(cleaned_text, pdf_path.name)
    
    # Create processed document
    document = {
        "metadata": metadata,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "char_count_raw": len(raw_text),
        "char_count_cleaned": len(cleaned_text),
    }
    
    # Save to processed directory
    output_dir = get_processed_data_dir()
    output_name = pdf_path.stem + ".json"
    output_path = output_dir / output_name
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    logger.info(
        f"Processed: {pdf_path.name} -> {output_name} "
        f"({document['char_count_cleaned']} chars)"
    )
    
    return document


def process_all_pdfs() -> dict:
    """
    Process all PDF files in the raw data directory.
    
    Returns:
        Dictionary mapping filenames to processing status.
    """
    raw_dir = get_raw_data_dir()
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    logger.info("=" * 60)
    logger.info("Starting PDF Text Extraction and Cleaning")
    logger.info(f"Source directory: {raw_dir}")
    logger.info(f"Output directory: {get_processed_data_dir()}")
    logger.info(f"PDFs to process: {len(pdf_files)}")
    logger.info("=" * 60)
    
    results = {}
    
    for pdf_path in pdf_files:
        doc = process_pdf(pdf_path)
        results[pdf_path.name] = doc is not None
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary:")
    for filename, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {filename}: {status}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    process_all_pdfs()
