#!/usr/bin/env python3
"""
Script to train the document classifier using synthetic data.
"""
import time
import datetime
import argparse
from typing import Optional

from src.utils.ml.dataset_manager import dataset_manager
from src.utils.ml.ml_classifier import DocumentClassifier


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    return str(datetime.timedelta(seconds=int(seconds)))


def print_dataset_stats(train_df, test_df):
    """Print statistics about the training and test datasets."""
    print("\n=== Dataset Statistics ===")
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    print("\nDistribution by document type:")
    
    train_dist = train_df["doc_type"].value_counts()
    test_dist = test_df["doc_type"].value_counts()
    
    for doc_type in sorted(train_dist.index):
        print(f"  {doc_type}: {train_dist.get(doc_type, 0)} train, {test_dist.get(doc_type, 0)} test")


def test_classifier(classifier, test_df):
    """Test the classifier on sample documents."""
    print("\n=== Testing Classifier ===")
    for i in range(min(5, len(test_df))):
        sample = test_df.iloc[i]
        prediction = classifier.predict(sample['text'])
        
        print(f"\nSample {i+1}:")
        print(f"  Actual: {sample['doc_type']}")
        print(f"  Predicted: {prediction['doc_type']}")
        print(f"  Confidence: {prediction['confidence']:.2f}")
        
        # Show top 3 probabilities
        top_3 = sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print("  Top probabilities:")
        for doc_type, prob in top_3:
            print(f"    {doc_type}: {prob:.2f}")


def main():
    """Train the document classifier using existing data."""
    parser = argparse.ArgumentParser(description='Train document classifier')
    parser.add_argument('--optimize', action='store_true', help='Optimize hyperparameters')
    args = parser.parse_args()
    
    # Load datasets
    try:
        print("Loading datasets...")
        train_df, test_df = dataset_manager.load_datasets()
        
        if train_df.empty or test_df.empty:
            print("Error: Empty datasets. Generate training data first with /generate_training_data endpoint.")
            return
    except FileNotFoundError:
        print("Error: No datasets found. Generate training data first with /generate_training_data endpoint.")
        return
    
    # Print dataset statistics
    print_dataset_stats(train_df, test_df)
    
    # Load and test the classifier
    print("\n=== Loading Classifier ===")
    training_start_time = time.time()
    print(f"Starting at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    classifier = DocumentClassifier()
    if not classifier.load_model():
        print("Error: Could not load pre-trained model.")
        return
    
    # Calculate total time
    training_time = time.time() - training_start_time
    print(f"\nModel loaded in {format_time(training_time)}")
    
    # Test the classifier
    test_classifier(classifier, test_df)


if __name__ == "__main__":
    main()
