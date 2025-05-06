"""
ML-based document classifier using TF-IDF and SVM.
"""
import os
import re
import tempfile
import numpy as np
from typing import Dict, Union

from src.utils import gcs_utils

# ML libraries
import joblib

# NLP preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Setup model directories and paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Default model file paths
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'document_classifier.joblib')
DEFAULT_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
DEFAULT_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')

class DocumentClassifier:
    """ML-based document classifier using TF-IDF vectorization and SVM."""
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.classes = None
        self.label_encoder = None
        self.is_trained = False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        if not text:
            return ""
        
        # Basic preprocessing: lowercase, remove special chars and digits
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def predict(self, text: str, preprocess: bool = True) -> Dict[str, Union[str, float]]:
        """Predict document type for given text."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Load a model first.")
        
        # Process text
        processed_text = self.preprocess_text(text) if preprocess else text
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        predicted_class_idx = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = probabilities[np.where(self.classifier.classes_ == predicted_class_idx)][0]
        
        # Handle label encoding
        if self.label_encoder is not None:
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            class_indices = self.classifier.classes_
            original_classes = self.label_encoder.inverse_transform(class_indices)
            prob_dict = {str(cls): float(prob) for cls, prob in zip(original_classes, probabilities)}
        else:
            predicted_class = str(predicted_class_idx)
            prob_dict = {str(cls): float(prob) for cls, prob in zip(self.classifier.classes_, probabilities)}
        
        return {
            "doc_type": predicted_class,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }

    def load_model(self, model_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH, 
                 label_encoder_path=DEFAULT_LABEL_ENCODER_PATH, force_gcs_load=False) -> bool:
        """Load a trained model from disk or GCS."""
        try:
            # First try to load from local filesystem (unless force_gcs_load is True)
            if not force_gcs_load and os.path.exists(model_path) and os.path.exists(vectorizer_path):
                print(f"Loading ML classifier model from local filesystem: {model_path}")
                self.classifier = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                
                # Load label encoder if available locally
                if os.path.exists(label_encoder_path):
                    self.label_encoder = joblib.load(label_encoder_path)
                    print(f"Loaded label encoder from {label_encoder_path}")
                else:
                    print(f"Warning: Label encoder not found at {label_encoder_path}")
                    self.label_encoder = None
            else:
                # If not found locally, try loading from GCS
                print("Local model files not found, trying to load from GCS...")
                
                # Create temporary directory for downloaded model files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Hardcoded GCS bucket and paths
                    GCS_BUCKET = "document-classifier-data-document-classifier-project"
                    model_gcs_path = "models/document_classifier.joblib"
                    vectorizer_gcs_path = "models/tfidf_vectorizer.joblib"
                    label_encoder_gcs_path = "models/label_encoder.joblib"
                    
                    # Override the default bucket name in gcs_utils for this operation
                    original_bucket = gcs_utils.DEFAULT_BUCKET_NAME
                    gcs_utils.DEFAULT_BUCKET_NAME = GCS_BUCKET
                    
                    try:
                        # Download model files from GCS
                        temp_model_path = os.path.join(temp_dir, "document_classifier.joblib")
                        temp_vectorizer_path = os.path.join(temp_dir, "tfidf_vectorizer.joblib")
                        temp_label_encoder_path = os.path.join(temp_dir, "label_encoder.joblib")
                        
                        # Check if files exist in GCS and download them
                        if gcs_utils.file_exists(model_gcs_path) and gcs_utils.file_exists(vectorizer_gcs_path):
                            print(f"Loading ML classifier model from GCS: {model_gcs_path}")
                            gcs_utils.download_file(model_gcs_path, temp_model_path)
                            gcs_utils.download_file(vectorizer_gcs_path, temp_vectorizer_path)
                            
                            # Load the downloaded files
                            self.classifier = joblib.load(temp_model_path)
                            self.vectorizer = joblib.load(temp_vectorizer_path)
                            
                            # Try to download and load label encoder
                            if gcs_utils.file_exists(label_encoder_gcs_path):
                                gcs_utils.download_file(label_encoder_gcs_path, temp_label_encoder_path)
                                self.label_encoder = joblib.load(temp_label_encoder_path)
                                print(f"Loaded label encoder from GCS: {label_encoder_gcs_path}")
                            else:
                                print(f"Warning: Label encoder not found in GCS")
                                self.label_encoder = None
                            
                            # Save the downloaded files to local filesystem for future use
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            joblib.dump(self.classifier, model_path)
                            joblib.dump(self.vectorizer, vectorizer_path)
                            if self.label_encoder is not None:
                                joblib.dump(self.label_encoder, label_encoder_path)
                        else:
                            print(f"Warning: Model files not found in GCS")
                            return False
                    finally:
                        # Always restore the original bucket name
                        gcs_utils.DEFAULT_BUCKET_NAME = original_bucket
            
            # Set classes and trained flag
            self.classes = self.classifier.classes_
            self.is_trained = True
            print("ML classifier model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def save_model(self, model_path: str = DEFAULT_MODEL_PATH,
                  vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
                  label_encoder_path: str = DEFAULT_LABEL_ENCODER_PATH) -> Dict[str, str]:
        """Save model components to disk."""
        if not self.is_trained:
            raise ValueError("Model is not trained. Nothing to save.")
            
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, label_encoder_path)
            
        return {
            "model_path": model_path,
            "vectorizer_path": vectorizer_path,
            "label_encoder_path": label_encoder_path if self.label_encoder else None
        }
