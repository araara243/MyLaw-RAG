"""
Semantic Chunking for Malaysian Legal Documents

This module implements structure-aware chunking for Malaysian legal texts.
Instead of naive token-based splitting, it chunks by legal sections to
ensure each chunk contains a complete legal definition or provision.

Key features:
- Detect Section headers (Section 1, Section 2, etc.)
- Detect Part headers (PART I, PART II, etc.)
- Keep subsections with their parent sections
- Include metadata (Act name, section number) for citation
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import tiktoken

from config import (
    RAGConfig,
    get_processed_dir,
    setup_logging
)

# Configure logging
logger = setup_logging(__name__)

# Token counting
TOKENIZER: Optional[Any] = None


def get_tokenizer() -> Any:
    """Lazy load the tokenizer."""
    global TOKENIZER
    if TOKENIZER is None:
        try:
            TOKENIZER = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    return TOKENIZER


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        return len(get_tokenizer().encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens, returning 0: {e}")
        return 0


@dataclass
class LegalChunk:
    """A chunk of legal text with metadata."""
    chunk_id: str
    act_name: str
    act_number: int
    part: Optional[str]
    section_number: Optional[str]
    section_title: Optional[str]
    content: str
    token_count: int
    start_position: int  # Character position in original text


# Regex patterns for Malaysian legal document structure
SECTION_PATTERN = re.compile(
    r"^(?:(?:Section|Seksyen)\s+(\d+[A-Za-z]*)|(\d+[A-Za-z]*)\.)\s*(.*)$",
    re.IGNORECASE | re.MULTILINE
)

PART_PATTERN = re.compile(
    r"^(?:PART|BAHAGIAN)\s+([IVXLCDM]+|\d+)\b[\s\-–—]*(.*)$",
    re.IGNORECASE | re.MULTILINE
)

SUBSECTION_PATTERN = re.compile(
    r"^\s*\((\d+[a-z]?)\)\s*",
    re.MULTILINE
)


def find_sections(text: str) -> List[Dict[str, Any]]:
    """
    Find all sections in the text with their positions.
    Returns all matches found by regex, sorted by position.
    Structure-aware chunking (in chunk_document) handles merging small chunks 
    (like TOC entries) and preserving large ones (Body sections).
    
    Returns a list of dicts with keys:
        - section_number: str
        - title: str
        - start: int (character position)
        - end: int (character position, exclusive)
    """
    sections = []
    
    for match in SECTION_PATTERN.finditer(text):
        # Group 1 is for "Section X", Group 2 is for "X."
        sec_num = match.group(1) or match.group(2)
        title = match.group(3).strip() if match.group(3) else ""
        
        if not sec_num:
            continue
            
        sections.append({
            "section_number": sec_num,
            "title": title,
            "start": match.start(),
            "end": None,  # Will be filled later
        })
    
    # Sort by start position
    sections.sort(key=lambda x: x["start"])
    
    # Set end positions
    for i, section in enumerate(sections):
        if i + 1 < len(sections):
            section["end"] = sections[i + 1]["start"]
        else:
            section["end"] = len(text)
    
    return sections


def find_current_part(text: str, position: int) -> Optional[str]:
    """
    Find the current PART header at a given position.
    
    Args:
        text: Full document text.
        position: Character position to check.
    
    Returns:
        Part identifier (e.g., "PART I - PRELIMINARY") or None.
    """
    current_part = None
    
    for match in PART_PATTERN.finditer(text):
        if match.start() <= position:
            part_num = match.group(1)
            part_title = match.group(2).strip() if match.group(2) else ""
            current_part = f"Part {part_num}"
            if part_title:
                current_part += f" - {part_title}"
        else:
            break
    
    return current_part


def split_large_section(
    section_text: str,
    max_tokens: int = 1000
) -> List[str]:
    """
    Split a large section into smaller chunks while preserving subsection boundaries.
    
    Args:
        section_text: The text of a single section.
        max_tokens: Maximum tokens per chunk.
    
    Returns:
        List of text chunks.
    """
    tokens = count_tokens(section_text)
    
    if tokens <= max_tokens:
        return [section_text]
    
    # Try splitting on subsection markers
    subsection_matches = list(SUBSECTION_PATTERN.finditer(section_text))
    
    if not subsection_matches:
        # No subsections, split by paragraphs
        paragraphs = section_text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            if count_tokens(test_chunk) > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks or [section_text]
    
    # Split on subsection boundaries
    chunks = []
    current_chunk = section_text[:subsection_matches[0].start()]
    
    for i, match in enumerate(subsection_matches):
        if i + 1 < len(subsection_matches):
            subsection_text = section_text[match.start():subsection_matches[i + 1].start()]
        else:
            subsection_text = section_text[match.start():]
        
        test_chunk = current_chunk + subsection_text
        
        if count_tokens(test_chunk) > max_tokens and current_chunk.strip():
            chunks.append(current_chunk.strip())
            current_chunk = subsection_text
        else:
            current_chunk = test_chunk
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks or [section_text]


def chunk_document(
    document: Dict[str, Any],
    max_tokens: int = 1000,
    min_tokens: int = 50
) -> List[LegalChunk]:
    """
    Chunk a processed legal document into semantic chunks.
    
    Args:
        document: A processed document dict with keys:
            - metadata: dict with act_name, act_number, etc.
            - cleaned_text: str
        max_tokens: Maximum tokens per chunk.
        min_tokens: Minimum tokens per chunk (smaller chunks merged).
    
    Returns:
        List of LegalChunk objects.
    """
    text = document.get("cleaned_text", "")
    metadata = document.get("metadata", {})
    act_name = metadata.get("act_name", "Unknown Act")
    act_number = metadata.get("act_number", 0)
    
    chunks: List[LegalChunk] = []
    sections = find_sections(text)
    
    if not sections:
        # No sections found, create a single chunk
        logger.warning(f"No sections found in {act_name}, creating single chunk")
        chunk = LegalChunk(
            chunk_id=f"act_{act_number}_full",
            act_name=act_name,
            act_number=act_number,
            part=None,
            section_number=None,
            section_title=None,
            content=text,
            token_count=count_tokens(text),
            start_position=0
        )
        return [chunk]
    
    # Add preamble if any text before first section
    if sections[0]["start"] > 0:
        preamble = text[:sections[0]["start"]].strip()
        if preamble and count_tokens(preamble) >= min_tokens:
            chunk = LegalChunk(
                chunk_id=f"act_{act_number}_preamble",
                act_name=act_name,
                act_number=act_number,
                part=find_current_part(text, 0),
                section_number="Preamble",
                section_title="Preliminary Provisions",
                content=preamble,
                token_count=count_tokens(preamble),
                start_position=0
            )
            chunks.append(chunk)
    
    # Process each section
    seen_ids = {} # dictionary to track counts of each base chunk_id

    for section in sections:
        section_text = text[section["start"]:section["end"]].strip()
        current_part = find_current_part(text, section["start"])
        
        # Split large sections
        section_chunks = split_large_section(section_text, max_tokens)
        
        for i, chunk_text in enumerate(section_chunks):
            if count_tokens(chunk_text) < min_tokens and chunks:
                # Merge small chunk with previous
                prev = chunks[-1]
                chunks[-1] = LegalChunk(
                    chunk_id=prev.chunk_id,
                    act_name=prev.act_name,
                    act_number=prev.act_number,
                    part=prev.part,
                    section_number=prev.section_number,
                    section_title=prev.section_title,
                    content=prev.content + "\n\n" + chunk_text,
                    token_count=count_tokens(prev.content + "\n\n" + chunk_text),
                    start_position=prev.start_position
                )
                continue
            
            # Base ID generation
            base_id = f"act_{act_number}_s{section['section_number']}"
            
            # Sub-chunk handling (from large section split)
            if len(section_chunks) > 1:
                base_id += f"_{i+1}"
            
            # Deduplication suffix handling
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                chunk_id = f"{base_id}_dup{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                chunk_id = base_id
            
            chunk = LegalChunk(
                chunk_id=chunk_id,
                act_name=act_name,
                act_number=act_number,
                part=current_part,
                section_number=section["section_number"],
                section_title=section["title"],
                content=chunk_text,
                token_count=count_tokens(chunk_text),
                start_position=section["start"]
            )
            chunks.append(chunk)
    
    return chunks


def process_all_documents(max_tokens: int = 1000) -> Dict[str, Dict[str, int]]:
    """
    Process all documents in the processed directory and create chunks.
    
    Args:
        max_tokens: Maximum tokens per chunk.
    
    Returns:
        Dictionary mapping filenames to chunk statistics.
    """
    processed_dir = get_processed_dir()
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return {}
        
    json_files = list(processed_dir.glob("*.json"))
    
    # Filter out chunked files
    json_files = [f for f in json_files if not f.name.endswith("_chunks.json")]
    
    logger.info("=" * 60)
    logger.info("Starting Semantic Chunking")
    logger.info(f"Source directory: {processed_dir}")
    logger.info(f"Documents to process: {len(json_files)}")
    logger.info(f"Max tokens per chunk: {max_tokens}")
    logger.info("=" * 60)
    
    results: Dict[str, Dict[str, int]] = {}
    all_chunks = []
    
    for json_path in json_files:
        logger.info(f"\nProcessing: {json_path.name}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                document = json.load(f)
            
            chunks = chunk_document(document, max_tokens)
            all_chunks.extend(chunks)
            
            # Save chunks for this document
            chunks_path = json_path.with_name(json_path.stem + "_chunks.json")
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created {len(chunks)} chunks -> {chunks_path.name}")
            results[json_path.name] = {
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
                "avg_tokens": sum(c.token_count for c in chunks) // len(chunks) if chunks else 0
            }
        except Exception as e:
            logger.error(f"Failed to process {json_path.name}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Chunking Summary:")
    total_chunks = 0
    for filename, stats in results.items():
        logger.info(
            f"  {filename}: {stats['chunk_count']} chunks, "
            f"avg {stats['avg_tokens']} tokens"
        )
        total_chunks += stats['chunk_count']
    logger.info(f"\nTotal chunks created: {total_chunks}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    config = RAGConfig()
    process_all_documents(max_tokens=config.chunk_size)
