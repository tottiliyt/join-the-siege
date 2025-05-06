#!/usr/bin/env python3
"""
Startup script to download model files from GCS when the container starts.
"""
import os
import tempfile
import joblib
from src.utils import gcs_utils
from src.utils.ml.ml_classifier import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH, DEFAULT_LABEL_ENCODER_PATH

def download_model_files():
    """Download model files from GCS at startup."""
    print("Downloading model files from GCS at startup...")
    
    # Hardcoded GCS bucket name
    GCS_BUCKET = "document-classifier-data-document-classifier-project"
    
    # GCS paths
    model_gcs_path = "models/document_classifier.joblib"
    vectorizer_gcs_path = "models/tfidf_vectorizer.joblib"
    label_encoder_gcs_path = "models/label_encoder.joblib"
    
    # Override the default bucket name in gcs_utils for this operation
    original_bucket = gcs_utils.DEFAULT_BUCKET_NAME
    gcs_utils.DEFAULT_BUCKET_NAME = GCS_BUCKET
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
        
        # Check if files exist in GCS and download them
        if gcs_utils.file_exists(model_gcs_path) and gcs_utils.file_exists(vectorizer_gcs_path):
            print(f"Found model files in GCS bucket: {GCS_BUCKET}")
            
            # Download model files
            gcs_utils.download_file(model_gcs_path, DEFAULT_MODEL_PATH)
            gcs_utils.download_file(vectorizer_gcs_path, DEFAULT_VECTORIZER_PATH)
            
            # Try to download label encoder
            if gcs_utils.file_exists(label_encoder_gcs_path):
                gcs_utils.download_file(label_encoder_gcs_path, DEFAULT_LABEL_ENCODER_PATH)
                print(f"Downloaded label encoder from GCS: {label_encoder_gcs_path}")
            else:
                print(f"Warning: Label encoder not found in GCS")
            
            print("Successfully downloaded model files from GCS")
            return True
        else:
            print(f"Warning: Model files not found in GCS bucket: {GCS_BUCKET}")
            return False
    except Exception as e:
        print(f"Error downloading model files from GCS: {e}")
        return False
    finally:
        # Always restore the original bucket name
        gcs_utils.DEFAULT_BUCKET_NAME = original_bucket

if __name__ == "__main__":
    download_model_files()
