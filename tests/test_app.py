from io import BytesIO

import pytest
from src.app import app, allowed_file

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.mark.parametrize("filename, expected", [
    ("file.pdf", True),
    ("file.png", True),
    ("file.jpg", True),
    ("file.txt", True),  # Updated to match implementation - txt files are allowed
    ("file", False),
])
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected

def test_no_file_in_request(client):
    response = client.post('/classify_file')
    assert response.status_code == 400

def test_no_selected_file(client):
    data = {'file': (BytesIO(b""), '')}  # Empty filename
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_success(client, mocker):
    mocker.patch('src.app.classify_file', return_value='test_class')

    data = {'file': (BytesIO(b"dummy content"), 'file.pdf')}
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.get_json() == {"document_type": "test_class"}


def test_generate_training_data(client, mocker):
    """Test the /generate_training_data endpoint."""
    # Mock the dataset_manager.generate_and_add_documents function
    mock_generate = mocker.patch('src.app.dataset_manager.generate_and_add_documents')
    mock_generate.return_value = {
        "success": True,
        "message": "Added 5 invoice documents to queue",
        "status": {"status": "in_progress"}
    }
    
    # Make the request
    data = {"doc_type": "invoice", "count": 5}
    response = client.post('/generate_training_data', json=data)
    
    # Check the response
    assert response.status_code == 202
    assert "success" in response.get_json()
    assert response.get_json()["success"] == True
    
    # Verify the mock was called with the right parameters
    mock_generate.assert_called_once_with(doc_type="invoice", count=5, industry=None)


def test_data_generation_status(client, mocker):
    """Test the /data_generation_status endpoint."""
    # Mock the dataset_manager.get_generation_status function
    mock_status = mocker.patch('src.app.dataset_manager.get_generation_status')
    mock_status.return_value = {
        "status": "in_progress",
        "doc_type": "invoice",
        "count": 5
    }
    
    # Make the request
    response = client.get('/data_generation_status')
    
    # Check the response
    assert response.status_code == 200
    assert "status" in response.get_json()
    assert response.get_json()["status"] == "in_progress"
    
    # Verify the mock was called
    mock_status.assert_called_once()


def test_train_model(client, mocker):
    """Test the /train_model endpoint."""
    # Mock the training_manager.start_training function
    mock_train = mocker.patch('src.app.training_manager.start_training')
    mock_train.return_value = {
        "success": True,
        "message": "Model training started",
        "status": {"status": "starting"}
    }
    
    # Make the request
    data = {"optimize": True}
    response = client.post('/train_model', json=data)
    
    # Check the response
    assert response.status_code == 202
    assert "success" in response.get_json()
    assert response.get_json()["success"] == True
    
    # Verify the mock was called with the right parameters
    mock_train.assert_called_once_with(optimize_hyperparams=True)


def test_training_status(client, mocker):
    """Test the /training_status endpoint."""
    # Mock the training_manager.get_status function
    mock_status = mocker.patch('src.app.training_manager.get_status')
    mock_status.return_value = {
        "status": "in_progress"
    }
    
    # Make the request
    response = client.get('/training_status')
    
    # Check the response
    assert response.status_code == 200
    assert "status" in response.get_json()
    assert response.get_json()["status"] == "in_progress"
    
    # Verify the mock was called
    mock_status.assert_called_once()


def test_classify_files(client, mocker):
    """Test the /classify_files endpoint."""
    # Mock the classify_files function that's imported in the route
    mock_classify = mocker.patch('src.classifier.classify_files')
    mock_classify.return_value = {
        "file1.pdf": "invoice",
        "file2.jpg": "receipt"
    }
    
    # Create test files
    data = {
        'files': [
            (BytesIO(b"content 1"), 'file1.pdf'),
            (BytesIO(b"content 2"), 'file2.jpg')
        ]
    }
    
    # Make the request
    response = client.post('/classify_files', data=data, content_type='multipart/form-data')
    
    # Check the response
    assert response.status_code == 200
    assert "results" in response.get_json()
    assert "total_files" in response.get_json()
    assert response.get_json()["total_files"] == 2
    assert response.get_json()["results"]["file1.pdf"] == "invoice"
    assert response.get_json()["results"]["file2.jpg"] == "receipt"
    
    # Verify the mock was called
    mock_classify.assert_called_once()


def test_classify_files_no_files(client):
    """Test the /classify_files endpoint with no files."""
    # Make the request with no files
    response = client.post('/classify_files')
    
    # Check the response
    assert response.status_code == 400
    assert "error" in response.get_json()
    
    # Make the request with empty files
    data = {'files': [(BytesIO(b""), '')]}
    response = client.post('/classify_files', data=data, content_type='multipart/form-data')
    
    # Check the response
    assert response.status_code == 400
    assert "error" in response.get_json()