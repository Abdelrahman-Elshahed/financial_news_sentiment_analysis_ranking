import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture
def sample_news_data():
    """Sample news texts for testing"""
    return [
        "The company reported strong earnings and a record high stock price.",
        "The company reported terrible losses and the stock price crashed.",
        "The tech sector outperformed, lifting the Nasdaq to a new peak.",
        "Central bank maintains interest rates at current levels."
    ]

@pytest.fixture
def mock_tokenizer():
    """Create a simple tokenizer for testing"""
    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(["company reported strong earnings stock price tech sector nasdaq interest rates"])
    return tokenizer

@pytest.fixture
def mock_model():
    """Create a simple model for testing"""
    inputs = tf.keras.Input(shape=(100,))
    # Remove input_length parameter
    x = tf.keras.layers.Embedding(1000, 64)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    outputs = tf.keras.layers.Dense(2, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model