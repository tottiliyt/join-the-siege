import pytest
from unittest.mock import patch, MagicMock
from src.utils.ml.data_generator import generate_documents_batch
from src.utils.ml.dataset_manager import dataset_manager

def test_generate_documents_batch():
    """Test that document batch generation works correctly."""
    # Test with a small batch size
    results = generate_documents_batch("invoice", count=1)
    
    # Should return a list with one item
    assert isinstance(results, list)
    assert len(results) == 1
    
    # Each result should have the expected keys
    result = results[0]
    assert "success" in result
    assert "doc_type" in result
    assert result["doc_type"] == "invoice"

def test_generate_and_add_documents():
    """Test the generate_and_add_documents method."""
    # Call the method
    result = dataset_manager.generate_and_add_documents("invoice", count=1)
    
    # Check the result structure
    assert result["success"] == True
    assert "status" in result
    assert "message" in result
    
    # Check that the status is being updated
    status = dataset_manager.get_generation_status()
    assert status["doc_type"] == "invoice"
    assert status["count"] == 1

@patch('src.utils.ml.dataset_manager.dataset_manager.add_document')
@patch('src.utils.ml.dataset_manager.generate_documents_batch')
def test_document_generation_process(mock_generate, mock_add):
    """Test the document generation process."""
    # Mock the generate_documents_batch function
    mock_generate.return_value = [
        {"success": True, "text": "Test content", "doc_type": "invoice"}
    ]
    
    # Mock the add_document method
    mock_add.return_value = {"success": True}
    
    # Call the method
    dataset_manager._run_document_generation("invoice", 1)
    
    # Verify the mocks were called correctly
    mock_generate.assert_called_once()
    mock_add.assert_called_once()
    
    # Check the status
    status = dataset_manager.get_generation_status()
    assert status["status"] in ["completed", "in_progress"]
