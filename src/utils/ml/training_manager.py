"""
Manager for background model training processes.
"""
import os
import threading
import time
import datetime
import tempfile
import numpy as np
from typing import Dict, Union

from src.utils import gcs_utils

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import joblib

from src.utils.ml.dataset_manager import dataset_manager
from src.utils.ml.ml_classifier import DEFAULT_MODEL_PATH, DEFAULT_VECTORIZER_PATH, MODEL_DIR

class TrainingManager:
    """Manages background model training processes."""
    
    def __init__(self):
        self._training_thread = None
        self._training_status = {
            "status": "idle",
            "error": None
        }
        self._lock = threading.Lock()
    
    def start_training(self, optimize_hyperparams: bool = False) -> Dict[str, Union[bool, str]]:
        """Start a new training process in the background."""
        with self._lock:
            # Check if training is already in progress
            if self._training_thread and self._training_thread.is_alive():
                return {
                    "success": False,
                    "message": "Training already in progress",
                    "status": self._training_status
                }
            
            # Reset status
            self._training_status = {
                "status": "starting",
                "error": None
            }
            
            # Start training thread
            self._training_thread = threading.Thread(
                target=self._run_training,
                args=(optimize_hyperparams,),
                daemon=True
            )
            self._training_thread.start()
            
            return {
                "success": True,
                "message": "Model training started",
                "status": self._training_status
            }
    
    def get_status(self) -> Dict:
        """Get the current training status."""
        with self._lock:
            return self._training_status.copy()
    
    def _run_training(self, optimize_hyperparams: bool = False):
        """Run the actual training process."""
        try:
            print("Starting model training...")
            self._update_status("in_progress")
            
            # Load datasets
            train_df, test_df = dataset_manager.load_datasets()
            
            if train_df.empty or test_df.empty:
                self._update_status("failed", error="Training data is empty. Generate data first.")
                return
            
            print("Processing data...")
            
            # Prepare data
            X_train = train_df["text"].tolist()
            y_train = train_df["doc_type"].tolist()
            X_test = test_df["text"].tolist()
            y_test = test_df["doc_type"].tolist()
            
            # Encode labels
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train + y_test)  # Fit on all labels
            y_train = label_encoder.transform(y_train)
            y_test = label_encoder.transform(y_test)
            
            # Create vectorizer and transform data
            print("Extracting features...")
            vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                stop_words='english'
            )
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            # Train classifier
            print("Training model...")
            base_model = LinearSVC(C=1.0, dual=False, class_weight='balanced', max_iter=2000)
            
            # Determine CV folds based on minimum samples per class
            min_samples = min(np.bincount(y_train))
            cv_folds = max(2, min(3, min_samples // 2))
            
            # Create calibrated classifier for probability estimates
            model = CalibratedClassifierCV(estimator=base_model, cv=cv_folds, method='sigmoid')
            model.fit(X_train_tfidf, y_train)
            
            # Evaluate and save model
            print("Finalizing model...")
            accuracy = np.mean(model.predict(X_test_tfidf) == y_test)
            
            # Save model components locally
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(model, DEFAULT_MODEL_PATH)
            joblib.dump(vectorizer, DEFAULT_VECTORIZER_PATH)
            label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.joblib')
            joblib.dump(label_encoder, label_encoder_path)
            
            # Save model components to GCS
            print("Saving model to Google Cloud Storage...")
            gcs_utils.upload_file(DEFAULT_MODEL_PATH, "models/document_classifier.joblib")
            gcs_utils.upload_file(DEFAULT_VECTORIZER_PATH, "models/tfidf_vectorizer.joblib")
            gcs_utils.upload_file(label_encoder_path, "models/label_encoder.joblib")
            
            # Update final status
            self._update_status("completed", accuracy=float(accuracy))
            print(f"Training completed with accuracy: {accuracy:.4f}")
            
        except Exception as e:
            self._update_status("failed", error=str(e))
            print(f"Training error: {e}")
    
    def _update_status(self, status, error=None, accuracy=None):
        """Helper method to update training status with thread safety."""
        with self._lock:
            self._training_status["status"] = status
            
            if error:
                self._training_status["error"] = error
            if accuracy is not None:
                self._training_status["accuracy"] = accuracy


# Singleton instance
training_manager = TrainingManager()
