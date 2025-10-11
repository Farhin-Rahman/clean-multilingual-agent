# services/files.py - File processing and text extraction

import os
import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)

def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        path: Path to the PDF file
    
    Returns:
        Extracted text or error message if failed
    """
    if not os.path.exists(path):
        logger.warning(f"PDF file not found: {path}")
        return f"Error: PDF file not found at {path}"
    
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
        
        doc.close()
        
        full_text = "\n".join(text_parts)
        
        if full_text.strip():
            logger.info(f"PDF extraction successful for {path}")
            return full_text.strip()
        else:
            logger.warning(f"No text found in PDF {path}")
            return "No text content found in PDF"
            
    except Exception as e:
        logger.error(f"PDF extraction failed for {path}: {e}")
        return f"Error reading PDF: {str(e)}"

def extract_text_from_txt(path: str) -> str:
    """
    Extract text from a plain text file.
    
    Args:
        path: Path to the text file
    
    Returns:
        File content or error message if failed
    """
    if not os.path.exists(path):
        logger.warning(f"Text file not found: {path}")
        return f"Error: Text file not found at {path}"
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                if content.strip():
                    logger.info(f"Text file read successful for {path} with {encoding}")
                    return content.strip()
                    
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Could not decode text file {path} with any encoding")
        return "Error: Could not decode text file"
        
    except Exception as e:
        logger.error(f"Text file reading failed for {path}: {e}")
        return f"Error reading text file: {str(e)}"

def extract_text_from_file(path: str) -> str:
    """
    Extract text from a file based on its extension.
    
    Args:
        path: Path to the file
    
    Returns:
        Extracted text or error message
    """
    if not os.path.exists(path):
        return f"Error: File not found at {path}"
    
    _, ext = os.path.splitext(path.lower())
    
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    elif ext in ['.txt', '.md', '.csv']:
        return extract_text_from_txt(path)
    else:
        return f"Error: Unsupported file type {ext}"

def split_chunks(text: str, max_tokens: int = 400, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk (approximate)
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    # Simple word-based splitting (approximate token count)
    words = text.split()
    
    if len(words) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Move start position with overlap but guarantee progress
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, max_tokens - overlap)
        start = next_start
        
        # Break if we've reached the end
        if end == len(words):
            break
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Remove excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def get_file_info(path: str) -> dict:
    """
    Get basic information about a file.
    
    Args:
        path: Path to the file
    
    Returns:
        Dict with file information
    """
    if not os.path.exists(path):
        return {"error": "File not found"}
    
    try:
        stat = os.stat(path)
        _, ext = os.path.splitext(path)
        
        return {
            "name": os.path.basename(path),
            "extension": ext.lower(),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "is_readable": os.access(path, os.R_OK)
        }
        
    except Exception as e:
        return {"error": str(e)}

def is_supported_file(path: str) -> bool:
    """
    Check if a file type is supported for text extraction.
    
    Args:
        path: Path to the file
    
    Returns:
        True if file type is supported
    """
    supported_extensions = {'.pdf', '.txt', '.md', '.csv'}
    _, ext = os.path.splitext(path.lower())
    return ext in supported_extensions

def validate_file(path: str) -> tuple[bool, str]:
    """
    Validate if a file can be processed.
    
    Args:
        path: Path to the file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(path):
        return False, "File does not exist"
    
    if not os.access(path, os.R_OK):
        return False, "File is not readable"
    
    if not is_supported_file(path):
        return False, "File type not supported"
    
    file_info = get_file_info(path)
    if "error" in file_info:
        return False, file_info["error"]
    
    # Check file size (limit to 50MB)
    if file_info["size_mb"] > 50:
        return False, "File too large (max 50MB)"
    
    return True, ""
