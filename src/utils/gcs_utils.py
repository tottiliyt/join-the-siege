"""
Utility functions for interacting with Google Cloud Storage.
"""
import os
import io
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from google.cloud import storage
from google.cloud.exceptions import NotFound

# Default bucket name - can be overridden with environment variable
DEFAULT_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'document-classifier-data-606021919371')

# Flag to track if GCS is available
GCS_AVAILABLE = True

# Create a client only once, with graceful fallback for local development
try:
    storage_client = storage.Client()
except Exception as e:
    print(f"Warning: Could not initialize GCS client: {e}")
    print("Falling back to local storage for development environment")
    storage_client = None
    GCS_AVAILABLE = False

def get_or_create_bucket(bucket_name: str = DEFAULT_BUCKET_NAME) -> Optional[storage.Bucket]:
    """Get or create a GCS bucket with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        print("GCS not available, skipping bucket operations")
        return None
        
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except NotFound:
        print(f"Bucket {bucket_name} not found, creating...")
        bucket = storage_client.create_bucket(bucket_name, location="us-central1")
    except Exception as e:
        print(f"Error accessing GCS bucket: {e}")
        return None
    return bucket

def upload_file(local_path: str, gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> str:
    """Upload a file to GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        print(f"GCS not available, skipping upload of {local_path} to {gcs_path}")
        return local_path
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return local_path
        
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket_name}/{gcs_path}"
    except Exception as e:
        print(f"Error uploading file to GCS: {e}")
        return local_path

def download_file(gcs_path: str, local_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> str:
    """Download a file from GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        print(f"GCS not available, skipping download of {gcs_path} to {local_path}")
        return local_path if os.path.exists(local_path) else None
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return local_path if os.path.exists(local_path) else None
        
    try:
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            print(f"File {gcs_path} not found in GCS bucket {bucket_name}")
            return local_path if os.path.exists(local_path) else None
            
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading file from GCS: {e}")
        return local_path if os.path.exists(local_path) else None

def upload_bytes(data: bytes, gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> str:
    """Upload bytes to GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        print(f"GCS not available, skipping upload of bytes to {gcs_path}")
        return f"local://{gcs_path}"
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return f"local://{gcs_path}"
        
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(data)
        return f"gs://{bucket_name}/{gcs_path}"
    except Exception as e:
        print(f"Error uploading bytes to GCS: {e}")
        return f"local://{gcs_path}"

def download_bytes(gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> Optional[bytes]:
    """Download bytes from GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        print(f"GCS not available, skipping download of bytes from {gcs_path}")
        return None
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return None
        
    try:
        blob = bucket.blob(gcs_path)
        if blob.exists():
            return blob.download_as_bytes()
        return None
    except Exception as e:
        print(f"Error downloading bytes from GCS: {e}")
        return None

def upload_string(data: str, gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> str:
    """Upload a string to GCS."""
    return upload_bytes(data.encode('utf-8'), gcs_path, bucket_name)

def download_string(gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> Optional[str]:
    """Download a string from GCS."""
    data = download_bytes(gcs_path, bucket_name)
    return data.decode('utf-8') if data else None

def upload_json(data: Dict, gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> str:
    """Upload JSON data to GCS with graceful fallback for local development."""
    json_data = json.dumps(data).encode('utf-8')
    
    # If GCS is not available, save to local file
    if not GCS_AVAILABLE:
        local_path = os.path.join(os.getcwd(), gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(json_data)
        return local_path
        
    return upload_bytes(json_data, gcs_path, bucket_name)

def download_json(gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> Optional[Dict]:
    """Download JSON data from GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        # Try to load from local file
        local_path = os.path.join(os.getcwd(), gcs_path)
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                json_data = f.read()
                return json.loads(json_data.decode('utf-8'))
        return None
        
    json_data = download_bytes(gcs_path, bucket_name)
    if json_data:
        return json.loads(json_data.decode('utf-8'))
    return None

def upload_dataframe(df: pd.DataFrame, gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> str:
    """Upload a pandas DataFrame to GCS as CSV with graceful fallback for local development."""
    csv_data = df.to_csv(index=False).encode('utf-8')
    
    # If GCS is not available, save to local file
    if not GCS_AVAILABLE:
        local_path = os.path.join(os.getcwd(), gcs_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(csv_data)
        return local_path
        
    return upload_bytes(csv_data, gcs_path, bucket_name)

def download_dataframe(gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> Optional[pd.DataFrame]:
    """Download a pandas DataFrame from GCS."""
    csv_data = download_bytes(gcs_path, bucket_name)
    if csv_data:
        return pd.read_csv(io.BytesIO(csv_data))
    return None

def file_exists(gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> bool:
    """Check if a file exists in GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        # Check if file exists locally
        local_path = os.path.join(os.getcwd(), gcs_path)
        return os.path.exists(local_path)
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return False
        
    try:
        blob = bucket.blob(gcs_path)
        return blob.exists()
    except Exception as e:
        print(f"Error checking if file exists in GCS: {e}")
        return False

def list_files(prefix: str = "", bucket_name: str = DEFAULT_BUCKET_NAME) -> List[str]:
    """List files in GCS with a given prefix with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        # List files locally with the given prefix
        local_dir = os.path.join(os.getcwd(), prefix)
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            result = []
            for root, _, files in os.walk(local_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, os.getcwd())
                    result.append(rel_path)
            return result
        elif os.path.exists(os.path.dirname(os.path.join(os.getcwd(), prefix))):
            # If prefix is a file pattern, list matching files
            import glob
            matches = glob.glob(os.path.join(os.getcwd(), prefix))
            return [os.path.relpath(match, os.getcwd()) for match in matches]
        return []
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return []
        
    try:
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        print(f"Error listing files in GCS: {e}")
        return []

def delete_file(gcs_path: str, bucket_name: str = DEFAULT_BUCKET_NAME) -> bool:
    """Delete a file from GCS with graceful fallback for local development."""
    if not GCS_AVAILABLE:
        # Delete file locally
        local_path = os.path.join(os.getcwd(), gcs_path)
        if os.path.exists(local_path):
            os.remove(local_path)
            return True
        return False
        
    bucket = get_or_create_bucket(bucket_name)
    if not bucket:
        return False
        
    try:
        blob = bucket.blob(gcs_path)
        if blob.exists():
            blob.delete()
            return True
        return False
    except Exception as e:
        print(f"Error deleting file from GCS: {e}")
        return False
