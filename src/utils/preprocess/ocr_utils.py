from PIL import Image
import pytesseract
from werkzeug.datastructures import FileStorage

def extract_text_from_image(file: FileStorage) -> str:
    """Extract text from an image file (jpg, png) using OCR. Returns extracted text as a single string (lowercased)."""
    try:
        file.seek(0)
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image)
        return text.lower()
    except Exception as e:
        # Optionally log the error here
        return ""
