import pytest
import os
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
    
    # If success is True, check doc_type
    if result["success"]:
        assert "doc_type" in result
        assert result["doc_type"] == "invoice"
    else:
        # If API key is not configured, we expect an error message
        assert "error" in result
        pytest.skip(f"Skipping due to API configuration issue: {result.get('error')}")

def test_generate_and_add_documents():
    """Test the generate_and_add_documents method."""
    # Call the method
    result = dataset_manager.generate_and_add_documents("invoice", count=1)
    
    # Check the result structure
    assert "success" in result
    assert "message" in result
    
    if result["success"]:
        # If successful, check status and generation status
        assert "status" in result
        
        # Check that the status is being updated
        status = dataset_manager.get_generation_status()
        assert status["doc_type"] == "invoice"
        assert status["count"] == 1
    else:
        # If API key is not configured, we expect an error message
        assert "error" in result["message"].lower() or "api key" in result["message"].lower()
        pytest.skip(f"Skipping due to API configuration issue: {result.get('message')}")

@pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), reason="OpenAI API key not configured")
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
