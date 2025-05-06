import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import numpy as np

from src.classifier import classify_file, preprocess_file, get_ml_classifier, classify_files
from src.utils.ml.ml_classifier import DocumentClassifier

class MockFileStorage:
    """Mock class for FileStorage objects."""
    def __init__(self, content, filename):
        self.stream = BytesIO(content)
        self.filename = filename
        
    def seek(self, offset):
        self.stream.seek(offset)
        
    def read(self):
        return self.stream.read()

def test_preprocess_file():
    """Test that file preprocessing works for different file types."""
    # Test with a PDF file
    with patch('src.classifier.extract_text_from_pdf', return_value="PDF content"):
        pdf_file = MockFileStorage(b"pdf content", "test.pdf")
        result = preprocess_file(pdf_file)
        assert result == "PDF content"
    
    # Test with an image file
    with patch('src.classifier.extract_text_from_image', return_value="Image content"):
        image_file = MockFileStorage(b"image content", "test.jpg")
        result = preprocess_file(image_file)
        assert result == "Image content"
    
    # Test with a text file
    with patch('src.classifier.extract_text_from_txt', return_value="Text content"):
        text_file = MockFileStorage(b"text content", "test.txt")
        result = preprocess_file(text_file)
        assert result == "Text content"

@patch('src.classifier.get_ml_classifier')
@patch('src.classifier.preprocess_file')
def test_classify_file(mock_preprocess, mock_get_classifier):
    """Test that file classification works correctly."""
    # Mock the preprocessing function
    mock_preprocess.return_value = "Sample document content"
    
    # Mock the classifier
    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = {"doc_type": "invoice", "confidence": 0.95}
    mock_get_classifier.return_value = mock_classifier
    
    # Test with a sample file
    file = MockFileStorage(b"sample content", "test.pdf")
    result = classify_file(file)
    
    # Check the result
    assert result == "invoice"
    
    # Verify the mocks were called correctly
    mock_preprocess.assert_called_once_with(file)
    mock_classifier.predict.assert_called_once_with("Sample document content")

@patch('src.classifier.DocumentClassifier')
def test_get_ml_classifier(mock_classifier_class):
    """Test that the ML classifier can be loaded."""
    # Mock the classifier class
    mock_classifier_instance = MagicMock()
    mock_classifier_instance.load_model.return_value = True
    mock_classifier_class.return_value = mock_classifier_instance
    
    # Get the classifier
    classifier = get_ml_classifier()
    
    # Check the result
    assert classifier is not None
    
    # Verify the mocks were called correctly
    mock_classifier_instance.load_model.assert_called_once_with(force_gcs_load=True)

def test_classify_file_no_text():
    """Test classification when no text can be extracted."""
    # Test with a file that can't be processed
    with patch('src.classifier.preprocess_file', return_value=""):
        file = MockFileStorage(b"empty content", "test.unknown")
        result = classify_file(file)
        assert result == "unknown file"


def test_classify_files():
    """Test that multiple files can be classified correctly."""
    # Create test files
    file1 = MockFileStorage(b"content 1", "file1.pdf")
    file2 = MockFileStorage(b"content 2", "file2.jpg")
    file3 = MockFileStorage(b"content 3", "file3.docx")
    
    # Test with a list of files
    with patch('src.classifier.classify_file') as mock_classify:
        # Set up the mock to return different document types
        mock_classify.side_effect = ["invoice", "receipt", "contract"]
        
        result = classify_files([file1, file2, file3])
        
        # Check the results
        assert isinstance(result, dict)
        assert len(result) == 3
        assert result["file1.pdf"] == "invoice"
        assert result["file2.jpg"] == "receipt"
        assert result["file3.docx"] == "contract"
        
        # Verify the mock was called for each file
        assert mock_classify.call_count == 3
    
    # Test with a single file
    with patch('src.classifier.classify_file') as mock_classify:
        mock_classify.return_value = "invoice"
        
        # Mock the isinstance check to return True for FileStorage
        with patch('src.classifier.isinstance') as mock_isinstance:
            mock_isinstance.return_value = True
            
            result = classify_files(file1)
            assert isinstance(result, dict)
            assert len(result) == 1
            assert result["file1.pdf"] == "invoice"
            assert mock_classify.call_count == 1
