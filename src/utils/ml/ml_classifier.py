"""
ML-based document classifier using TF-IDF and SVM.
"""
import os
import re
import numpy as np
from typing import Dict, Union

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

    def load_model(self, model_path: str = DEFAULT_MODEL_PATH, 
                  vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
                  label_encoder_path: str = DEFAULT_LABEL_ENCODER_PATH) -> bool:
        """Load trained model components from disk."""
        try:
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.classes = self.classifier.classes_
            
            # Load label encoder if available
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                print(f"Loaded label encoder from {label_encoder_path}")
            else:
                print(f"Warning: Label encoder not found at {label_encoder_path}")
                self.label_encoder = None
            
            self.is_trained = True
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
