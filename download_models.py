#!/usr/bin/env python3
"""
Script to pre-download FastEmbed models during Docker build.
This ensures models are available immediately when the container starts.
"""

import os
from fastembed import TextEmbedding

def download_models():
    """Download the required FastEmbed models."""
    print("Pre-downloading FastEmbed models...")
    
    try:
        # Download the thenlper/gte-large model used in the application
        print("Downloading thenlper/gte-large model...")
        embedding_model = TextEmbedding(model_name="thenlper/gte-large")
        print("✓ thenlper/gte-large model downloaded successfully")
        
        # Test the model to ensure it's working
        test_text = ["This is a test sentence."]
        embeddings = list(embedding_model.embed(test_text))
        print(f"✓ Model test successful - embedding dimension: {len(embeddings[0])}")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        raise

if __name__ == "__main__":
    download_models()
    print("All models downloaded successfully!") 