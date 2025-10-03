"""
ONNX Model Conversion Script
This script handles the conversion of transformer models to ONNX format
for optimized inference in production environments.
"""

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer
)
import os
# Configuration
HF_TOKEN = os.getenv('HF_TOKEN')

def load_bias_comment_model():
    """Load the BERT-based model for bias comment classification."""
    model_path = "valurank/distilroberta-bias"
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=HF_TOKEN)
    return tokenizer, model

def load_toxic_comment_model():
    """Load the toxic comment classification model."""
    model_path = "martin-ha/toxic-comment-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, token=HF_TOKEN)
    return tokenizer, model

def load_bart_mnli_model():
    """Load the BART MNLI model for zero-shot classification."""
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

if __name__ == "__main__":
    # Load models
    bias_tokenizer, bias_model = load_bias_comment_model()
    toxic_tokenizer, toxic_model = load_toxic_comment_model()
    bart_tokenizer, bart_model = load_bart_mnli_model()
    
    print("Models loaded successfully!")
    print(f"Bias model: {bias_model.__class__.__name__}")
    print(f"Toxic model: {toxic_model.__class__.__name__}")
    print(f"BART model: {bart_model.__class__.__name__}")

