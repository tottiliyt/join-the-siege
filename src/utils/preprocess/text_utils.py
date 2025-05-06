"""
Utility functions for extracting text from plain text files.
"""
from werkzeug.datastructures import FileStorage
import chardet

def extract_text_from_txt(file: FileStorage) -> str:
    """
    Extract text from plain text files (.txt, .csv, .md, etc.).
    
    Args:
        file (FileStorage): The uploaded text file
        
    Returns:
        str: Extracted text as a single string (lowercased)
    """
    try:
        file.seek(0)
        # Read the file content as bytes
        content_bytes = file.read()
        
        # Detect encoding
        detected = chardet.detect(content_bytes)
        encoding = detected['encoding'] or 'utf-8'
        
        # Decode using detected encoding
        text = content_bytes.decode(encoding)
        
        return text.lower()
    except Exception as e:
        # Optionally log the error here
        return ""
