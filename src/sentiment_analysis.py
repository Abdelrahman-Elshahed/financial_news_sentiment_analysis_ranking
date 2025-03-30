import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def predict_sentiment(model, texts, tokenizer, max_length=100, model_type='lstm', binary_classification=False):

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    if model_type.lower() == 'lstm':
        # Tokenize and pad sequences
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        # Get predictions
        sentiment_scores = model.predict(padded_sequences)
    
    elif model_type.lower() == 'logreg':
        # For logistic regression, use the model pipeline
        sentiment_scores = model.predict(texts)
        
    elif model_type.lower() == 'bert':
        # For BERT models, use the tokenizer from transformers
        from transformers import BertTokenizer
        
        # Ensure tokenizer is the right type
        if not isinstance(tokenizer, BertTokenizer):
            raise ValueError("For BERT models, tokenizer must be a BertTokenizer")
            
        # Encode the texts
        encoded_input = tokenizer(
            texts, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length', 
            return_tensors='tf'
        )
        
        # Get predictions
        sentiment_scores = model.predict([encoded_input['input_ids'], encoded_input['attention_mask']])
        
        # For models with multiple outputs
        if isinstance(sentiment_scores, list) and len(sentiment_scores) > 1:
            sentiment_scores = sentiment_scores[0]  # Take the first output (sentiment)
            
    else:
        raise ValueError("Model type must be 'lstm', 'logreg', or 'bert'")
    
    # Handle binary classification
    if binary_classification:
        # Convert from binary classification format to the standard format
        # Assuming binary scores are [positive, negative]
        if sentiment_scores.shape[1] == 2:
            # Create a third column for neutral with zeros
            neutral_scores = np.zeros((sentiment_scores.shape[0], 1))
            sentiment_scores = np.hstack((sentiment_scores, neutral_scores))
    
    return sentiment_scores

def analyze_sentiment_distribution(sentiment_scores, texts=None):

    # Calculate average sentiment scores
    avg_pos = np.mean(sentiment_scores[:, 0])
    avg_neg = np.mean(sentiment_scores[:, 1])
    avg_neu = np.mean(sentiment_scores[:, 2])
    
    # Calculate dominant sentiment for each text
    dominant_sentiments = np.argmax(sentiment_scores, axis=1)
    sentiment_counts = {
        'Positive': np.sum(dominant_sentiments == 0),
        'Negative': np.sum(dominant_sentiments == 1),
        'Neutral': np.sum(dominant_sentiments == 2)
    }
    
    # Create plots if matplotlib is available
    try:
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot average sentiment scores
        labels = ['Positive', 'Negative', 'Neutral']
        values = [avg_pos, avg_neg, avg_neu]
        ax1.bar(labels, values, color=['green', 'red', 'gray'])
        ax1.set_title('Average Sentiment Scores')
        ax1.set_ylim([0, 1])
        
        # Plot sentiment distribution
        ax2.pie([sentiment_counts[label] for label in labels], 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=['green', 'red', 'gray'])
        ax2.set_title('Dominant Sentiment Distribution')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/sentiment_distribution.png')
        plt.close()
        
        plot_path = 'plots/sentiment_distribution.png'
    except:
        plot_path = None
    
    results = {
        'average_scores': {
            'positive': avg_pos,
            'negative': avg_neg,
            'neutral': avg_neu
        },
        'dominant_sentiment_counts': sentiment_counts,
        'plot_path': plot_path
    }
    
    # If texts are provided, add examples of each sentiment
    if texts is not None:
        examples = {}
        for i, label in enumerate(['Positive', 'Negative', 'Neutral']):
            # Find indices of top 3 examples for this sentiment
            indices = np.argsort(sentiment_scores[:, i])[-3:]
            examples[label] = [texts[idx] for idx in indices]
        
        results['examples'] = examples
    
    return results

def classify_sentiment(sentiment_scores, binary=False):

    if binary and sentiment_scores.shape[1] >= 2:
        # For binary sentiment, just compare positive vs negative
        positive_scores = sentiment_scores[:, 0]
        negative_scores = sentiment_scores[:, 1]
        classifications = ['POSITIVE' if pos > neg else 'NEGATIVE' 
                          for pos, neg in zip(positive_scores, negative_scores)]
    else:
        # Find the highest sentiment score for each text
        dominant_indices = np.argmax(sentiment_scores, axis=1)
        
        # Map indices to sentiment labels
        label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
        classifications = [label_map[idx] for idx in dominant_indices]
    
    return classifications

def plot_training_history(history):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_history.png')
    plt.close()
    
    return 'plots/training_history.png'

def plot_confusion_matrices(y_true, y_pred, labels=['Positive', 'Negative', 'Neutral']):

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
    
    for i, label in enumerate(labels):
        if len(labels) > 1:
            ax = axes[i]
        else:
            ax = axes
            
        y_true_binary = (y_true[:, i] > 0.5).astype(int)
        y_pred_binary = (y_pred[:, i] > 0.5).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title(f'{label} Sentiment')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/confusion_matrices.png')
    plt.close()
    
    return 'plots/confusion_matrices.png'

def test_model_examples(model, tokenizer, max_length=100, model_type='lstm'):

    # Example texts
    example_news = [
        "Stock market reaches all-time high as tech companies report strong earnings",
        "Major company announces layoffs amid economic downturn",
        "Central bank maintains current interest rates",
        "Company reported strong earnings and a record high stock price",
        "The CEO resigned unexpectedly, and the stock price tumbled"
    ]
    
    # Get predictions
    predictions = predict_sentiment(model, example_news, tokenizer, max_length, model_type)
    
    # Format results
    results = []
    for text, pred in zip(example_news, predictions):
        if pred.shape[0] >= 3:  # 3-class model
            results.append({
                "text": text,
                "positive_score": float(pred[0]),
                "negative_score": float(pred[1]),
                "neutral_score": float(pred[2]) if pred.shape[0] > 2 else 0.0,
                "dominant_sentiment": ["Positive", "Negative", "Neutral"][np.argmax(pred)]
            })
        else:  # Binary model
            results.append({
                "text": text,
                "positive_score": float(pred[0]),
                "negative_score": float(pred[1]),
                "neutral_score": 0.0,
                "dominant_sentiment": "Positive" if pred[0] > pred[1] else "Negative"
            })
    
    return results
