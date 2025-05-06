from PyPDF2 import PdfReader
from werkzeug.datastructures import FileStorage

def extract_text_from_pdf(file: FileStorage) -> str:
    """Extract all text from a PDF file. Returns extracted text as a single string (lowercased)."""
    try:
        file.seek(0)
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.lower()
    except Exception as e:
        # Optionally log the error here
        return ""
