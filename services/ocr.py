# services/ocr.py - OCR text extraction from images

import os
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # load .env from project root

logger = logging.getLogger("services.ocr")

try:
    import pytesseract
    # 1) Use env var if present
    cmd = os.getenv("TESSERACT_CMD")
    if cmd and os.path.exists(cmd):
        pytesseract.pytesseract.tesseract_cmd = cmd
        logger.info(f"Using Tesseract at: {cmd}")
    else:
        # 2) Try to find it on PATH or common Windows locations
        from shutil import which
        found = which("tesseract")
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
            logger.info(f"Using Tesseract (PATH): {found}")
        else:
            for candidate in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ):
                if os.path.exists(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    logger.info(f"Using Tesseract (fallback): {candidate}")
                    break
except Exception:
    pytesseract = None


def extract_text_from_image(path: str, lang: str = "eng") -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        path: Path to the image file
        lang: Language code for OCR (default: 'eng')
    
    Returns:
        Extracted text or empty string if failed
    """
    if not os.path.exists(path):
        logger.warning(f"Image file not found: {path}")
        return ""
    
    # Try pytesseract first
    try:
        import pytesseract
        from PIL import Image
        
        image = Image.open(path)
        text = pytesseract.image_to_string(image, lang=lang)
        
        if text.strip():
            logger.info(f"OCR successful with pytesseract for {path}")
            return text.strip()
        
    except ImportError:
        logger.info("pytesseract not available, trying OCR.space")
    except Exception as e:
        logger.warning(f"pytesseract failed for {path}: {e}")
    
    # Fallback to OCR.space API
    try:
        api_key = os.getenv("OCR_SPACE_API_KEY")
        if not api_key:
            logger.warning("OCR_SPACE_API_KEY not configured")
            return ""
        
        with open(path, 'rb') as image_file:
            files = {'file': image_file}
            data = {
                'apikey': api_key,
                'language': lang,
                'isOverlayRequired': False,
                'OCREngine': 2  # Use OCR Engine 2 for better accuracy
            }
            
            response = requests.post(
                'https://api.ocr.space/parse/image',
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('IsErroredOnProcessing', False):
                    logger.warning(f"OCR.space processing error: {result.get('ErrorMessage', 'Unknown error')}")
                    return ""
                
                parsed_results = result.get('ParsedResults', [])
                if parsed_results:
                    text = parsed_results[0].get('ParsedText', '')
                    if text.strip():
                        logger.info(f"OCR successful with OCR.space for {path}")
                        return text.strip()
            
            logger.warning(f"OCR.space API returned status {response.status_code}")
            
    except Exception as e:
        logger.warning(f"OCR.space failed for {path}: {e}")
    
    logger.warning(f"All OCR methods failed for {path}")
    return ""

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
        from PIL import Image
        
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            text = extract_text_from_image(tmp_path, lang)
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
