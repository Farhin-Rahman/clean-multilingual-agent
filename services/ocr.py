# services/ocr.py
from __future__ import annotations
import os, io, logging, shutil, requests
from typing import Optional
from PIL import Image  # pillow is already in requirements
import streamlit as st

logger = logging.getLogger(__name__)

def _get_tesseract_cmd() -> Optional[str]:
    # Respect explicit override, then PATH
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        return cmd
    return shutil.which("tesseract")

def ocr_image(path: str) -> str:
    """
    Returns extracted text from an image file.
    Strategy:
      1) pytesseract if a tesseract binary is available.
      2) OCR.Space API if configured.
      3) else return "" (and caller will handle gracefully).
    """
    text = ""
    try:
        tess = _get_tesseract_cmd()
        if tess:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tess
            text = pytesseract.image_to_string(Image.open(path)) or ""
            if text.strip():
                logger.info("OCR via pytesseract succeeded")
                return text
            logger.warning("pytesseract returned empty text; will try OCR.Space if available.")
    except Exception as e:
        logger.warning("pytesseract path/method unavailable: %s", e)

    # --- Fallback: OCR.Space ---
    # Avoid accessing st.secrets here to prevent errors when no secrets.toml exists
    key = os.getenv("OCR_SPACE_API_KEY")
    if key:
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    "https://api.ocr.space/parse/image",
                    data={"apikey": key, "OCREngine": 2, "scale": True, "language": "eng"},
                    files={"file": (os.path.basename(path), f)},
                    timeout=60,
                )
            resp.raise_for_status()
            data = resp.json()
            parsed = data.get("ParsedResults", [{}])[0].get("ParsedText", "")
            if parsed and parsed.strip():
                logger.info("OCR via OCR.Space succeeded")
                return parsed
            logger.warning("OCR.Space returned no text.")
        except Exception as e:
            logger.warning("OCR.Space failed: %s", e)

    logger.warning("No OCR available; returning empty text.")
    return ""

# Legacy function names for backward compatibility
def extract_text_from_image(path: str, lang: str = "eng") -> str:
    """Legacy function name - calls ocr_image."""
    return ocr_image(path)

def extract_text_from_image_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Extract text from image bytes using OCR.
    
    Args:
        image_bytes: Image data as bytes
        lang: Language code for OCR
    
    Returns:
        Extracted text or empty string if failed
    """
    try:
        import tempfile
        
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            text = ocr_image(tmp_path)
            return text
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
                
    except Exception as e:
        logger.warning(f"Failed to extract text from image bytes: {e}")
        return ""

def is_image_file(path: str) -> bool:
    """
    Check if a file is an image based on extension.
    
    Args:
        path: File path to check
    
    Returns:
        True if file appears to be an image
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    _, ext = os.path.splitext(path.lower())
    return ext in image_extensions

def get_supported_languages() -> list:
    """
    Get list of supported OCR languages.
    
    Returns:
        List of language codes
    """
    # Common languages supported by both pytesseract and OCR.space
    return [
        'eng', 'spa', 'fra', 'deu', 'ita', 'por', 'rus', 'chi_sim', 'chi_tra',
        'jpn', 'kor', 'ara', 'hin', 'ben', 'tam', 'tel', 'mar', 'guj', 'kan',
        'mal', 'ori', 'pan', 'pun', 'sin', 'urd'
    ]

def validate_image_file(path: str) -> tuple[bool, str]:
    """
    Validate if an image file can be processed.
    
    Args:
        path: Path to image file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(path):
        return False, "File does not exist"
    
    if not is_image_file(path):
        return False, "File is not a supported image format"
    
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"