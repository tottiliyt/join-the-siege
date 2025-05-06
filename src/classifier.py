from werkzeug.datastructures import FileStorage
import os
import mimetypes

# Import all text extraction utilities
from src.utils.preprocess.pdf_utils import extract_text_from_pdf
from src.utils.preprocess.ocr_utils import extract_text_from_image
from src.utils.preprocess.docx_utils import extract_text_from_docx
from src.utils.preprocess.excel_utils import extract_text_from_excel
from src.utils.preprocess.json_utils import extract_text_from_json
from src.utils.preprocess.text_utils import extract_text_from_txt

# Import ML classifier
from src.utils.ml.ml_classifier import DocumentClassifier

# Define file type mappings
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
DOCUMENT_EXTENSIONS = {'pdf', 'docx', 'doc'}
SPREADSHEET_EXTENSIONS = {'xlsx', 'xls', 'csv'}
TEXT_EXTENSIONS = {'txt', 'md'}
JSON_EXTENSIONS = {'json', 'json5'}

def preprocess_file(file: FileStorage) -> str:
    """
    Extract text from various file formats.
    
    Args:
        file (FileStorage): The uploaded file
        
    Returns:
        str: Extracted text content or empty string if extraction failed
    """
    if not file or not file.filename:
        return ""
    
    # Get file extension
    filename = file.filename.lower()
    ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''
    
    # Reset file pointer to beginning
    file.seek(0)
    
    # Extract text based on file type
    try:
        # PDF documents
        if ext == 'pdf':
            return extract_text_from_pdf(file)
        
        # Images (using OCR)
        elif ext in IMAGE_EXTENSIONS:
            return extract_text_from_image(file)
        
        # Word documents
        elif ext == 'docx' or ext == 'doc':
            return extract_text_from_docx(file)
        
        # Excel spreadsheets
        elif ext in SPREADSHEET_EXTENSIONS:
            return extract_text_from_excel(file)
        
        # JSON files
        elif ext in JSON_EXTENSIONS:
            return extract_text_from_json(file)
        
        # Plain text files
        elif ext in TEXT_EXTENSIONS:
            return extract_text_from_txt(file)
        
        # Unsupported file type
        else:
            # Try to guess based on content if extension is missing or unknown
            return "unknown file"
    
    except Exception as e:
        # Log the error here in production
        print(f"Error extracting text from {filename}: {str(e)}")
        return ""

# Initialize the ML classifier (lazy loading - will only load when first used)
_ml_classifier = None
_model_last_modified = 0

def get_ml_classifier():
    """Get or initialize the ML classifier."""
    global _ml_classifier
    
    # Always create a fresh instance for each request to ensure we're using the latest model
    try:
        print("Creating fresh ML classifier instance for this request")
        _ml_classifier = DocumentClassifier()
        
        # Force load from GCS by setting force_gcs_load=True
        # This will bypass local file checks and always attempt to load from GCS
        success = _ml_classifier.load_model(force_gcs_load=True)
        
        if success:
            print("ML classifier model loaded successfully from GCS")
        else:
            print("Warning: Failed to load ML classifier model from GCS")
            _ml_classifier = None
    except Exception as e:
        print(f"Warning: Could not load ML classifier: {e}")
        _ml_classifier = None
    
    return _ml_classifier

def classify_file(file: FileStorage):
    """Classify a file based on its content using ML."""
    # Extract text using the preprocessing pipeline
    text = preprocess_file(file)
    
    # If no text could be extracted, return unknown
    if not text:
        return "unknown file"
    
    # Try to use the ML classifier
    classifier = get_ml_classifier()
    if classifier:
        try:
            result = classifier.predict(text)
            
            # Convert all values to Python native types to ensure JSON serializability
            if isinstance(result, dict):
                # Ensure the document type is a string
                doc_type = str(result.get('doc_type', 'unknown file'))
                return doc_type
            else:
                # If result is not a dictionary, convert it to string
                return str(result)
        except Exception as e:
            print(f"ML classification error: {e}")
    
    # Fallback: return unknown
    return "unknown file"


def classify_files(files):
    """Classify multiple files based on their content using ML.
    
    Args:
        files: List of FileStorage objects or a FileStorage object
        
    Returns:
        dict: Dictionary mapping filenames to document types
    """
    # Handle single file case
    if isinstance(files, FileStorage):
        return {files.filename: classify_file(files)}
    
    # Process multiple files
    results = {}
    for file in files:
        if file and file.filename:
            results[file.filename] = classify_file(file)
    
    return results
