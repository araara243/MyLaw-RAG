"""
AGC Malaysia Legal Acts Scraper

This module downloads Malaysian legal acts (PDFs) from the Attorney General's
Chambers (AGC) official website: https://lom.agc.gov.my/

Target acts for MVP:
- Contracts Act 1950 (Act 136)
- Specific Relief Act 1951 (Act 137)
- Housing Development (Control and Licensing) Act 1966 (Act 118)
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://lom.agc.gov.my"
PDF_BASE_EN = f"{BASE_URL}/ilims/upload/portal/akta/LOM/EN"
PDF_BASE_BM = f"{BASE_URL}/ilims/upload/portal/akta/LOM/BM"
ACT_DETAIL_URL = f"{BASE_URL}/act-detail.php"

# MVP Acts to download
MVP_ACTS = [
    {"act_no": 136, "name": "Contracts Act 1950"},
    {"act_no": 137, "name": "Specific Relief Act 1951"},
    {"act_no": 118, "name": "Housing Development (Control and Licensing) Act 1966"},
]

# Request headers to mimic browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_raw_data_dir() -> Path:
    """Get the raw data directory, creating it if necessary."""
    raw_dir = get_project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def download_pdf(url: str, output_path: Path, retries: int = 3) -> bool:
    """
    Download a PDF file from a URL.
    
    Args:
        url: The URL of the PDF to download.
        output_path: The path to save the downloaded PDF.
        retries: Number of retry attempts on failure.
    
    Returns:
        True if download was successful, False otherwise.
    """
    for attempt in range(retries):
        try:
            logger.info(f"Downloading: {url}")
            response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            
            if response.status_code == 200:
                # Verify it's actually a PDF
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
                    logger.warning(f"Response is not a PDF: {content_type}")
                    return False
                
                # Write to file
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Saved: {output_path}")
                return True
            
            elif response.status_code == 404:
                logger.warning(f"PDF not found (404): {url}")
                return False
            
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{retries} failed: "
                    f"HTTP {response.status_code}"
                )
        
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1}/{retries} error: {e}")
        
        # Wait before retry
        if attempt < retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return False


def construct_pdf_url(act_no: int, language: str = "EN") -> str:
    """
    Construct the direct PDF URL for an Act.
    
    The AGC website uses a consistent URL pattern for the latest version:
    - English: /ilims/upload/portal/akta/LOM/EN/Act {act_no}.pdf
    - Malay: /ilims/upload/portal/akta/LOM/BM/Akta {act_no}.pdf
    
    Args:
        act_no: The Act number.
        language: "EN" for English, "BM" for Bahasa Malaysia.
    
    Returns:
        The constructed PDF URL.
    """
    if language.upper() == "EN":
        filename = f"Act {act_no}.pdf"
        base = PDF_BASE_EN
    else:
        filename = f"Akta {act_no}.pdf"
        base = PDF_BASE_BM
    
    # URL encode the filename (spaces become %20)
    encoded_filename = quote(filename)
    return f"{base}/{encoded_filename}"


def scrape_pdf_url_from_page(act_no: int, language: str = "BI") -> Optional[str]:
    """
    Scrape the PDF URL from the Act detail page.
    
    This is a fallback method if the direct URL pattern doesn't work.
    The page uses JavaScript to set the PDF source, so this method
    looks for the $src variable in the page scripts.
    
    Args:
        act_no: The Act number.
        language: "BI" for English Interface, "BM" for Malay.
    
    Returns:
        The PDF URL if found, None otherwise.
    """
    url = f"{ACT_DETAIL_URL}?language={language}&act={act_no}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        # Look for the $src variable in the page
        # Pattern: $src = "https://lom.agc.gov.my/ilims/upload/portal/akta/..."
        match = re.search(r'\$src\s*=\s*["\']([^"\']+\.pdf)["\']', response.text)
        
        if match:
            pdf_url = match.group(1)
            logger.info(f"Found PDF URL in page: {pdf_url}")
            return pdf_url
        
        # Alternative: look for iframe src
        soup = BeautifulSoup(response.text, "html.parser")
        iframe = soup.find("iframe")
        if iframe and iframe.get("src", "").endswith(".pdf"):
            return iframe["src"]
        
        logger.warning(f"Could not find PDF URL in page for Act {act_no}")
        return None
    
    except requests.RequestException as e:
        logger.error(f"Error scraping page for Act {act_no}: {e}")
        return None


def download_act(act_no: int, act_name: str, language: str = "EN") -> bool:
    """
    Download a specific Act PDF.
    
    Args:
        act_no: The Act number.
        act_name: The name of the Act (for filename).
        language: "EN" for English, "BM" for Bahasa Malaysia.
    
    Returns:
        True if download was successful, False otherwise.
    """
    output_dir = get_raw_data_dir()
    
    # Clean filename
    safe_name = re.sub(r'[<>:"/\\|?*]', '', act_name)
    filename = f"Act_{act_no}_{safe_name}_{language}.pdf"
    output_path = output_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        logger.info(f"Already exists: {output_path}")
        return True
    
    # Try direct URL first
    pdf_url = construct_pdf_url(act_no, language)
    if download_pdf(pdf_url, output_path):
        return True
    
    # Fallback: scrape URL from page
    logger.info(f"Direct URL failed, trying page scrape for Act {act_no}")
    scrape_lang = "BI" if language == "EN" else "BM"
    pdf_url = scrape_pdf_url_from_page(act_no, scrape_lang)
    
    if pdf_url:
        return download_pdf(pdf_url, output_path)
    
    logger.error(f"Failed to download Act {act_no}: {act_name}")
    return False


def download_mvp_acts() -> dict:
    """
    Download all MVP Acts defined in the module.
    
    Returns:
        A dictionary with act numbers as keys and download status as values.
    """
    results = {}
    
    logger.info("=" * 60)
    logger.info("Starting AGC Legal Acts Scraper")
    logger.info(f"Target directory: {get_raw_data_dir()}")
    logger.info(f"Acts to download: {len(MVP_ACTS)}")
    logger.info("=" * 60)
    
    for act in MVP_ACTS:
        act_no = act["act_no"]
        act_name = act["name"]
        
        logger.info(f"\nProcessing: Act {act_no} - {act_name}")
        
        # Download English version
        en_success = download_act(act_no, act_name, "EN")
        results[f"Act_{act_no}_EN"] = en_success
        
        # Small delay between requests to be polite
        time.sleep(1)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Download Summary:")
    for act_key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {act_key}: {status}")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    download_mvp_acts()
