import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import joblib

from src.utils.ml.training_manager import training_manager

def test_start_training():
    """Test that the training process can be started."""
    # Start training
    result = training_manager.start_training(optimize_hyperparams=False)
    
    # Check the result
    assert result["success"] == True
    assert "status" in result
    assert result["status"]["status"] in ["starting", "in_progress"]
    
    # Get the status
    status = training_manager.get_status()
    assert status["status"] in ["starting", "in_progress"]

@patch('src.utils.ml.training_manager.TrainingManager._run_training')
def test_run_training(mock_run_training):
    """Test the training process with a mocked _run_training method."""
    # Reset the training status to idle
    with training_manager._lock:
        training_manager._training_status = {
            "status": "idle",
            "error": None
        }
        training_manager._training_thread = None
    
    # Start training
    result = training_manager.start_training(optimize_hyperparams=False)
    
    # Check the result
    assert result["success"] == True
    assert "status" in result
    
    # Verify the mock was called
    mock_run_training.assert_called_once_with(False)

def test_get_status():
    """Test that the status can be retrieved."""
    # Get the status
    status = training_manager.get_status()
    
    # Check the status object
    assert isinstance(status, dict)
    assert "status" in status
    assert status["status"] in ["idle", "starting", "in_progress", "completed", "failed"]
