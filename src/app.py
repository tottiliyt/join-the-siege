from flask import Flask, request, jsonify
import os
import uuid
from datetime import datetime

from src.classifier import classify_file
from src.utils.ml.data_generator import DOCUMENT_TYPES
from src.utils.ml.dataset_manager import dataset_manager
from src.utils.ml.training_manager import training_manager

app = Flask(__name__)

# Import file extension sets from classifier
from src.classifier import IMAGE_EXTENSIONS, DOCUMENT_EXTENSIONS, SPREADSHEET_EXTENSIONS, TEXT_EXTENSIONS, JSON_EXTENSIONS

# Combine all supported extensions
ALLOWED_EXTENSIONS = set().union(
    IMAGE_EXTENSIONS,       # png, jpg, jpeg, gif, etc.
    DOCUMENT_EXTENSIONS,    # pdf, docx, doc
    SPREADSHEET_EXTENSIONS, # xlsx, xls, csv, etc.
    TEXT_EXTENSIONS,        # txt, md, html, etc.
    JSON_EXTENSIONS         # json, json5
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/classify_file', methods=['POST'])
def classify_file_route():
    """Endpoint to classify an uploaded file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed"}), 400

    document_type = classify_file(file)
    
    # Format the response
    response = {
        "document_type": document_type
    }
    
    return jsonify(response), 200


@app.route('/classify_files', methods=['POST'])
def classify_files_route():
    """Endpoint to classify multiple uploaded files."""
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400
    
    # Filter out files with disallowed extensions
    valid_files = [file for file in files if allowed_file(file.filename)]
    if not valid_files:
        return jsonify({"error": "No valid files found"}), 400
    
    # Classify all valid files
    from src.classifier import classify_files
    results = classify_files(valid_files)
    
    # Format the response
    response = {
        "results": results,
        "total_files": len(valid_files),
        "invalid_files": len(files) - len(valid_files)
    }
    
    return jsonify(response), 200


@app.route('/generate_training_data', methods=['POST'])
def generate_training_data_route():
    """Generate training data for one or more document types"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Accept either a single doc_type string or a list of doc_types
    doc_types = data.get('doc_types')
    if not doc_types:
        doc_type = data.get('doc_type')  # For backward compatibility
        if not doc_type:
            return jsonify({"error": "doc_type or doc_types is required"}), 400
        doc_types = [doc_type]
    elif isinstance(doc_types, str):
        doc_types = [doc_types]
    
    # Get other parameters
    samples_per_type = data.get('samples_per_type', data.get('count', 10))  # Default to 10 documents
    industry = data.get('industry')  # Optional industry context
    
    # Validate document types
    invalid_types = [dt for dt in doc_types if dt not in DOCUMENT_TYPES]
    if invalid_types:
        return jsonify({
            "error": f"Invalid document type(s): {', '.join(invalid_types)}",
            "valid_types": DOCUMENT_TYPES
        }), 400
    
    # Start asynchronous data generation for each document type
    results = []
    for doc_type in doc_types:
        result = dataset_manager.generate_and_add_documents(
            doc_type=doc_type,
            count=samples_per_type,
            industry=industry
        )
        results.append(result)
    
    # Return combined result immediately
    overall_success = all(r["success"] for r in results)
    response = {
        "success": overall_success,
        "message": f"Started generating {samples_per_type} samples for each of {len(doc_types)} document types",
        "doc_types": doc_types,
        "samples_per_type": samples_per_type,
        "results": results
    }
    
    if overall_success:
        return jsonify(response), 202  # 202 Accepted - request accepted for processing
    else:
        return jsonify(response), 400  # 400 Bad Request

@app.route('/data_generation_status', methods=['GET'])
def data_generation_status_route():
    """Get the status of the current data generation process"""
    status = dataset_manager.get_generation_status()
    return jsonify(status), 200

@app.route('/train_model', methods=['POST'])
def train_model_route():
    """Endpoint to train the document classifier model."""
    # Get parameters from request
    data = request.json or {}
    optimize = data.get('optimize', False)
    
    # Start training in a background thread
    result = training_manager.start_training(optimize_hyperparams=optimize)
    
    # Return result
    if result["success"]:
        return jsonify(result), 202  # 202 Accepted - request accepted for processing
    else:
        return jsonify(result), 400

@app.route('/training_status', methods=['GET'])
def training_status_route():
    """Get the status of the current model training process"""
    status = training_manager.get_status()
    return jsonify(status), 200


if __name__ == '__main__':
    app.run(debug=True)
