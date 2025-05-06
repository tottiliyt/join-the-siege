"""
Streamlined dataset management for document classification.
"""
import os
import json
import datetime
import uuid
import threading
import random
import tempfile
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

from src.utils import gcs_utils

from src.utils.ml.data_generator import DOCUMENT_TYPES, generate_documents_batch

# Setup data directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
ML_DATA_DIR = os.path.join(PROJECT_ROOT, 'ml_data')
TRAIN_DATA_DIR = os.path.join(ML_DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(ML_DATA_DIR, 'test')

# Create directories
for directory in [ML_DATA_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


class DatasetManager:
    """Manages document classification datasets and generation."""
    
    def __init__(self):
        self._generation_thread = None
        self._generation_queue = []
        self._generation_status = {
            "status": "idle",
            "error": None,
            "doc_type": None,
            "count": 0
        }
        self._lock = threading.Lock()
    
    @staticmethod
    def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test datasets, or create empty ones if they don't exist."""
        # Hardcoded GCS bucket name
        GCS_BUCKET = "document-classifier-data-document-classifier-project"
        
        # GCS paths for datasets
        train_gcs_path = "ml_data/train_data.csv"
        test_gcs_path = "ml_data/test_data.csv"
        
        columns = ["doc_type", "text", "industry", "id", "timestamp"]
        
        # Override the default bucket name in gcs_utils for this operation
        original_bucket = gcs_utils.DEFAULT_BUCKET_NAME
        gcs_utils.DEFAULT_BUCKET_NAME = GCS_BUCKET
        
        try:
            # Try to load from GCS
            train_df = gcs_utils.download_dataframe(train_gcs_path)
            test_df = gcs_utils.download_dataframe(test_gcs_path)
            
            # If not found, create empty DataFrames
            if train_df is None:
                train_df = pd.DataFrame(columns=columns)
            if test_df is None:
                test_df = pd.DataFrame(columns=columns)
                
            # Also try loading from local filesystem as fallback
            if train_df.empty and test_df.empty:
                train_csv_path = os.path.join(ML_DATA_DIR, "train_data.csv")
                test_csv_path = os.path.join(ML_DATA_DIR, "test_data.csv")
                
                if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
                    train_df = pd.read_csv(train_csv_path)
                    test_df = pd.read_csv(test_csv_path)
                    
                    # Upload to GCS for future use
                    gcs_utils.upload_dataframe(train_df, train_gcs_path)
                    gcs_utils.upload_dataframe(test_df, test_gcs_path)
        finally:
            # Always restore the original bucket name
            gcs_utils.DEFAULT_BUCKET_NAME = original_bucket
        
        return train_df, test_df
    
    def save_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, str]:
        """Save datasets to CSV and individual text files."""
        # Hardcoded GCS bucket name
        GCS_BUCKET = "document-classifier-data-document-classifier-project"
        
        # GCS paths for datasets
        train_gcs_path = "ml_data/train_data.csv"
        test_gcs_path = "ml_data/test_data.csv"
        stats_gcs_path = "ml_data/dataset_stats.json"
        
        print(f"Saving {len(train_df)} training and {len(test_df)} test documents")
        
        # Override the default bucket name in gcs_utils for this operation
        original_bucket = gcs_utils.DEFAULT_BUCKET_NAME
        gcs_utils.DEFAULT_BUCKET_NAME = GCS_BUCKET
        
        try:
            # Save to GCS
            gcs_utils.upload_dataframe(train_df, train_gcs_path)
            gcs_utils.upload_dataframe(test_df, test_gcs_path)
            
            # Save stats
            stats = {
                "train_distribution": train_df["doc_type"].value_counts().to_dict(),
                "test_distribution": test_df["doc_type"].value_counts().to_dict(),
                "total_samples": len(train_df) + len(test_df),
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            gcs_utils.upload_json(stats, stats_gcs_path)
        finally:
            # Always restore the original bucket name
            gcs_utils.DEFAULT_BUCKET_NAME = original_bucket
        
        # Also save to local filesystem for backward compatibility
        train_csv_path = os.path.join(ML_DATA_DIR, "train_data.csv")
        test_csv_path = os.path.join(ML_DATA_DIR, "test_data.csv")
        stats_path = os.path.join(ML_DATA_DIR, "dataset_stats.json")
        
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save individual documents to GCS
        def save_documents(df, base_dir, gcs_prefix):
            for _, row in df.iterrows():
                doc_type = row["doc_type"]
                doc_id = row.get("id", str(uuid.uuid4()))
                
                # Save locally
                type_dir = os.path.join(base_dir, doc_type)
                os.makedirs(type_dir, exist_ok=True)
                
                file_path = os.path.join(type_dir, f"{doc_id}.txt")
                if not os.path.exists(file_path):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(row["text"])
                
                # Save to GCS
                gcs_path = f"{gcs_prefix}/{doc_type}/{doc_id}.txt"
                gcs_utils.upload_string(row["text"], gcs_path)
        
        save_documents(train_df, TRAIN_DATA_DIR, "ml_data/train")
        save_documents(test_df, TEST_DATA_DIR, "ml_data/test")
        
        return {
            "train_csv": f"gs://{gcs_utils.DEFAULT_BUCKET_NAME}/{train_gcs_path}", 
            "test_csv": f"gs://{gcs_utils.DEFAULT_BUCKET_NAME}/{test_gcs_path}", 
            "stats": f"gs://{gcs_utils.DEFAULT_BUCKET_NAME}/{stats_gcs_path}"
        }
    
    def get_generation_status(self) -> Dict:
        """Get the current document generation status."""
        with self._lock:
            return self._generation_status.copy()
    
    def add_document(self, doc_type: str, text: str, industry: Optional[str] = None) -> Dict[str, Union[bool, str]]:
        """Add a new document to the dataset (80% training, 20% test)."""
        try:
            # Load existing datasets
            train_df, test_df = self.load_datasets()
            
            # Create new document entry
            new_doc = {
                'doc_type': doc_type,
                'text': text,
                'industry': industry or 'general',
                'id': str(uuid.uuid4()),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Add to training or test set
            is_training = random.random() < 0.8
            target_df = train_df if is_training else test_df
            target_df = pd.concat([target_df, pd.DataFrame([new_doc])], ignore_index=True)
            
            if is_training:
                train_df = target_df
                dataset_type = "training"
            else:
                test_df = target_df
                dataset_type = "test"
            
            # Save updated datasets
            self.save_datasets(train_df, test_df)
            
            return {
                "success": True,
                "message": f"Document added to {dataset_type} dataset",
                "train_count": len(train_df),
                "test_count": len(test_df)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to add document: {str(e)}"}
    
    def generate_and_add_documents(self, doc_type: str, count: int = 1, industry: Optional[str] = None) -> Dict[str, Union[bool, str, Dict]]:
        """Start asynchronous generation and addition of documents to the dataset."""
        # Validate document type
        if doc_type not in DOCUMENT_TYPES:
            return {
                "success": False,
                "message": f"Invalid document type: {doc_type}",
                "valid_types": DOCUMENT_TYPES
            }
        
        # Add to the generation queue
        with self._lock:
            self._generation_queue.append({
                "doc_type": doc_type,
                "count": count,
                "industry": industry
            })
            
            queue_size = len(self._generation_queue)
            
            # Update status
            if self._generation_status["status"] == "idle":
                self._generation_status.update({
                    "status": "starting",
                    "doc_type": doc_type,
                    "count": count
                })
            
            # Start the generation thread if needed
            if self._generation_thread is None or not self._generation_thread.is_alive():
                self._generation_thread = threading.Thread(
                    target=self._process_generation_queue,
                    daemon=True
                )
                self._generation_thread.start()
            
            return {
                "success": True,
                "message": f"Added {count} {doc_type} documents to queue (position {queue_size})",
                "status": self._generation_status
            }
    
    def _process_generation_queue(self) -> None:
        """Process the document generation queue."""
        print(f"Processing queue with {len(self._generation_queue)} items")
        
        try:
            while True:
                with self._lock:
                    if not self._generation_queue:
                        print("Queue empty, processing complete")
                        self._generation_status["status"] = "completed"
                        break
                    
                    next_item = self._generation_queue.pop(0)
                    self._generation_status.update({
                        "doc_type": next_item["doc_type"],
                        "count": next_item["count"],
                        "status": "in_progress"
                    })
                
                # Process the item
                print(f"Processing: {next_item['doc_type']} ({next_item['count']} samples)")
                self._run_document_generation(
                    next_item["doc_type"], 
                    next_item["count"], 
                    next_item["industry"]
                )
        
        except Exception as e:
            print(f"Queue processing error: {e}")
            with self._lock:
                self._generation_status.update({
                    "status": "failed",
                    "error": f"Queue error: {str(e)}"
                })
                    
    def _run_document_generation(self, doc_type: str, count: int, industry: Optional[str] = None) -> None:
        """Generate documents and add them to the dataset."""
        try:
            # Generate documents in batches
            documents_added = 0
            batch_size = min(5, count)
            
            for i in range(0, count, batch_size):
                current_batch_size = min(batch_size, count - i)
                batch_results = generate_documents_batch(doc_type, current_batch_size, industry)
                
                # Add successful generations
                for result in batch_results:
                    if result["success"]:
                        self.add_document(doc_type, result["text"], industry)
                        documents_added += 1
            
            # Update status
            with self._lock:
                self._generation_status["status"] = "completed"
        
        except Exception as e:
            with self._lock:
                self._generation_status.update({
                    "status": "failed",
                    "error": str(e)
                })
    
    def get_dataset_statistics(self) -> Dict[str, Union[int, Dict[str, int]]]:
        """Get statistics about the current datasets."""
        try:
            train_df, test_df = self.load_datasets()
            return {
                "train_count": len(train_df),
                "test_count": len(test_df),
                "total_count": len(train_df) + len(test_df)
            }
        except Exception as e:
            return {"error": f"Failed to get statistics: {str(e)}"}


# Create a singleton instance for easy access
dataset_manager = DatasetManager()
