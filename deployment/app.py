from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Optional
import tensorflow as tf
import numpy as np
import pickle
import re
import unidecode
import os
import nltk
from nltk.corpus import stopwords

# Initialize FastAPI app
app = FastAPI(
    title="News Sentiment Analysis API",
    description="API for analyzing sentiment in financial news",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure NLTK resources are downloaded
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Define preprocessing functions
def preprocess_text(text):
    """Lowercase the text and remove special characters, numbers, and punctuation"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = unidecode.unidecode(text)  # Remove accents/diacritics
    return text

def tokenize_and_remove_stopwords(text):
    """Tokenize text (split by spaces) and remove stopwords"""
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Model and tokenizer paths
MODEL_PATH = "/app/models/lstm_bert_model.h5"  # Absolute path in Docker container
TOKENIZER_PATH = "/app/models/tokenizer.pkl"   # Absolute path in Docker container

# Optional: Add fallback to relative paths for local development
import os
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "../models/lstm_bert_model.h5"
if not os.path.exists(TOKENIZER_PATH):
    TOKENIZER_PATH = "../models/tokenizer.pkl"

# Global variables for model and tokenizer
model = None
tokenizer = None

# Define request models
class NewsRequest(BaseModel):
    text: str

class NewsListRequest(BaseModel):
    texts: List[str]

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer when the app starts"""
    global model, tokenizer
    
    try:
        # Load tokenizer
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Define custom_objects to help with model loading
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError()
        }
        
        # Load the model
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Model and tokenizer loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        # We'll let the app start anyway and return errors on requests

def predict_news_sentiment(text, max_length=100):
    """
    Predicts binary sentiment (positive or negative) for news text
    """
    if not text or text.isspace():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    cleaned_text = tokenize_and_remove_stopwords(cleaned_text)
    
    # Convert to sequence and pad
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Get the vocab size from the model's embedding layer
    model_vocab_size = model.layers[0].input_dim
    
    # Filter out any word indices that are too large for the embedding layer
    filtered_sequence = [[idx for idx in seq if idx < model_vocab_size] for seq in sequence]
    
    padded_sequence = pad_sequences(
        filtered_sequence, maxlen=max_length, padding='post'
    )
    
    # Perform inference
    output = model.predict(padded_sequence, verbose=0)
    
    # Extract scores (binary model: [positive, negative])
    positive_score = float(output[0][0])
    negative_score = float(output[0][1])
    
    # Determine the sentiment (binary: positive or negative only)
    sentiment_label = "POSITIVE" if positive_score >= negative_score else "NEGATIVE"
    sentiment_score = positive_score if positive_score >= negative_score else negative_score
    
    # Calculate rank score (financial importance score)
    rank_score = (positive_score * 5) - (negative_score * 3)
    
    # Return full result dict for internal use
    return {
        "text": text,
        "sentiment": sentiment_label,
        "sentiment_score": float(sentiment_score),
        "positive_score": positive_score, 
        "negative_score": negative_score,
        "neutral_score": 0.0,
        "rank_score": float(rank_score)
    }

def filter_sentiment_result(full_result):
    """
    Filter out the fields we don't want to expose in the API
    and add 0.5 to the sentiment score
    """
    # Add 0.5 to sentiment_score (capping at 1.0 to keep it in a reasonable range)
    adjusted_score = min(1.0, full_result["sentiment_score"] + 0.5)
    
    # Only include the fields we want, explicitly removing the rest
    return {
        "text": full_result["text"],
        "sentiment": full_result["sentiment"],
        "sentiment_score": adjusted_score,
        "rank_score": full_result["rank_score"]
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the News Sentiment Analysis API"}

@app.post("/analyze")
async def analyze_sentiment(news: NewsRequest):
    """
    Analyze sentiment of a single news article
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")
    
    try:
        full_result = predict_news_sentiment(news.text)
        
        # Use the filter function to only return the fields we want
        return filter_sentiment_result(full_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.post("/analyze-batch")
async def analyze_batch(request: NewsListRequest):
    """
    Analyze sentiment of multiple news articles and rank them
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded")
    
    try:
        results = []
        for text in request.texts:
            full_result = predict_news_sentiment(text)
            
            # Filter the result to only include fields we want to expose
            filtered_result = filter_sentiment_result(full_result)
            results.append(filtered_result)
        
        # Sort results by rank_score (most positive first)
        results.sort(key=lambda x: x["rank_score"], reverse=True)
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing texts: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)