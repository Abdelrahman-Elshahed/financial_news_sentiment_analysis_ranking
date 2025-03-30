import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment_analysis import predict_sentiment

def test_predict_sentiment(mock_model, mock_tokenizer, sample_news_data):
    """Test sentiment prediction function"""
    results = predict_sentiment(
        model=mock_model,
        texts=sample_news_data,
        tokenizer=mock_tokenizer,
        binary_classification=True
    )
    
    # Check that results have the expected shape
    assert isinstance(results, np.ndarray)  # Changed from dict to np.ndarray
    # The predict_sentiment function returns an array, not a dictionary
    assert results.shape[1] >= 2  # Should have at least positive and negative columns
    assert len(results) == len(sample_news_data)