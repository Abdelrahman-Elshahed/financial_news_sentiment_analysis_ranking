import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from experiment_tracking import setup_mlflow, log_model_training

# Import local modules
from data_loader import load_dataset, convert_sentiment_to_binary, split_data, apply_min_max_scaling
from preprocessing import (
    preprocess_text, tokenize_and_remove_stopwords, 
    tokenize_and_pad_sequences, apply_preprocessing_pipeline, 
    create_bert_embedding_matrix_optimized
)
from model import build_lstm_model, build_logistic_regression_model
from train import (
    train_lstm_model, evaluate_lstm_model, 
    train_logistic_regression_model, evaluate_logistic_regression_model,
    evaluate_classification_metrics, save_models
)
from sentiment_analysis import predict_sentiment, analyze_sentiment_distribution, classify_sentiment
from ranking import rank_news_by_financial_importance, get_top_news, get_bottom_news, export_ranked_news

def process_and_analyze_news(news_texts, sentiment_scores=None, train_test_split_ratio=0.8, use_existing_models=False, 
                           lstm_model_path=None, lr_model_path=None):

    # Create output directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Split data into train and test sets
    X_train, X_test = train_test_split(news_texts, test_size=1-train_test_split_ratio, random_state=42)
    
    # Preprocess text (tokenization and padding)
    max_length = 100
    
    if use_existing_models:
        # Load tokenizer when using pre-trained models
        try:
            with open('../models/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            print("Loaded tokenizer from saved file")
            
            # Use loaded tokenizer for both train and test
            X_train_padded, _ = tokenize_and_pad_sequences(X_train, max_length, tokenizer=tokenizer, fit_on_texts=False)
            X_test_padded, _ = tokenize_and_pad_sequences(X_test, max_length, tokenizer=tokenizer, fit_on_texts=False)
        except FileNotFoundError:
            # If tokenizer file doesn't exist, create a new one
            print("Tokenizer file not found. Creating new tokenizer.")
            X_train_padded, tokenizer = tokenize_and_pad_sequences(X_train, max_length)
            X_test_padded, _ = tokenize_and_pad_sequences(X_test, max_length, tokenizer=tokenizer, fit_on_texts=False)
            
            # Save the tokenizer for future use
            with open('../models/tokenizer.pkl', 'wb') as f:
                pickle.dump(tokenizer, f)
    else:
        # Create new tokenizer
        X_train_padded, tokenizer = tokenize_and_pad_sequences(X_train, max_length)
        X_test_padded, _ = tokenize_and_pad_sequences(X_test, max_length, tokenizer=tokenizer, fit_on_texts=False)
        
        # Save the tokenizer
        with open('../models/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
    
    # Load or train LSTM model
    if use_existing_models and lstm_model_path and os.path.exists(lstm_model_path):
        # Define custom objects to handle the loading issue
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError()
        }
        
        try:
            # Try loading with custom objects
            lstm_model = tf.keras.models.load_model(
                lstm_model_path, 
                custom_objects=custom_objects,
                compile=False  # Load without compiling first
            )
            
            # Recompile the model with the correct loss and metrics
            lstm_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            print(f"Loaded LSTM model from {lstm_model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new LSTM model instead...")
            lstm_model = None
    else:
        lstm_model = None
    
    # Load or train Logistic Regression model
    if use_existing_models and lr_model_path and os.path.exists(lr_model_path):
        try:
            with open(lr_model_path, 'rb') as f:
                lr_model = pickle.load(f)
            print(f"Loaded Logistic Regression model from {lr_model_path}")
        except Exception as e:
            print(f"Error loading Logistic Regression model: {e}")
            lr_model = None
    else:
        lr_model = None
    
    # Get sentiment predictions
    if lstm_model is not None:
        sentiment_predictions = lstm_model.predict(X_test_padded)
    elif sentiment_scores is not None:
        sentiment_predictions = sentiment_scores
    else:
        raise ValueError("No model or sentiment scores available for prediction")
    
    # Analyze sentiment distribution
    sentiment_analysis = analyze_sentiment_distribution(sentiment_predictions, X_test)
    
    # Rank news by financial importance
    ranked_news = rank_news_by_financial_importance(X_test, sentiment_predictions)
    
    # Get top and bottom news
    top_news = get_top_news(ranked_news, n=10, include_scores=True)
    bottom_news = get_bottom_news(ranked_news, n=10, include_scores=True)
    
    return {
        'sentiment_predictions': sentiment_predictions,
        'sentiment_analysis': sentiment_analysis,
        'ranked_news': ranked_news,
        'top_news': top_news,
        'bottom_news': bottom_news,
        'lstm_model': lstm_model,
        'lr_model': lr_model,
        'tokenizer': tokenizer
    }

def train_new_models(df, save_dir='saved_models', use_bert=True):

    os.makedirs(save_dir, exist_ok=True)
    
    # Features (text) and targets 
    X = df["news"]
    y = df[["pos", "neg", "neu"]]
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Preprocess text
    X_train, X_val, X_test = apply_preprocessing_pipeline(X_train, X_val, X_test)
    
    # Tokenize and pad sequences
    max_length = 100
    X_train_padded, tokenizer = tokenize_and_pad_sequences(X_train, max_length)
    X_val_padded, _ = tokenize_and_pad_sequences(X_val, max_length, tokenizer=tokenizer, fit_on_texts=False)
    X_test_padded, _ = tokenize_and_pad_sequences(X_test, max_length, tokenizer=tokenizer, fit_on_texts=False)
    
    # Combine target variables for scaling
    y_train_combined = np.column_stack((y_train["pos"], y_train["neg"], y_train["neu"]))
    y_val_combined = np.column_stack((y_val["pos"], y_val["neg"], y_val["neu"]))
    y_test_combined = np.column_stack((y_test["pos"], y_test["neg"], y_test["neu"]))
    
    # Apply scaling
    y_train_scaled, y_val_scaled, y_test_scaled, target_scaler = apply_min_max_scaling(
        y_train_combined, y_val_combined, y_test_combined
    )
    # Then in your train_new_models function:
    mlflow = setup_mlflow()

    # Define parameters to track
    params = {
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "vocab_size": trunc_vocab_size,
        "use_bert": use_bert,
        "batch_size": 64,
        "epochs": 10
    }

    # After training
    log_model_training(
        model=lstm_model,
        history=lstm_history,
        metrics=lstm_metrics,
        params=params
    )
    # Get embedding matrix for LSTM if using BERT
    if use_bert:
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = 768  # BERT base hidden size
        max_words = 25000   # Maximum words to process
        
        try:
            # Try to load cached embeddings if available
            embedding_matrix_path = os.path.join(save_dir, 'bert_embedding_matrix.npy')
            if os.path.exists(embedding_matrix_path):
                embedding_matrix = np.load(embedding_matrix_path)
                print("Using cached BERT embedding matrix")
            else:
                print("Creating BERT embedding matrix (this may take a while)...")
                # Use optimized method to create embeddings
                embedding_matrix = create_bert_embedding_matrix_optimized(
                    tokenizer.word_index, 
                    max_words=max_words,
                    batch_size=64
                )
                # Save embeddings for future use
                np.save(embedding_matrix_path, embedding_matrix)
            
            # Use truncated vocabulary size to match embedding matrix
            trunc_vocab_size = min(vocab_size, max_words + 1)
        except Exception as e:
            print(f"Error creating BERT embeddings: {e}")
            print("Falling back to randomly initialized embeddings")
            embedding_matrix = None
            trunc_vocab_size = vocab_size
    else:
        embedding_matrix = None
        trunc_vocab_size = vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = 100
    
    # Build and train LSTM model
    print("Building and training LSTM model...")
    lstm_model = build_lstm_model(trunc_vocab_size, embedding_dim, max_length, embedding_matrix)
    lstm_history = train_lstm_model(
        lstm_model, 
        X_train_padded, 
        y_train_scaled, 
        X_val_padded, 
        y_val_scaled,
        epochs=10
    )
    
    # Evaluate LSTM model
    print("Evaluating LSTM model...")
    lstm_results = evaluate_lstm_model(lstm_model, X_test_padded, y_test_scaled)
    lstm_metrics = evaluate_classification_metrics(y_test_scaled, lstm_results['predictions'])
    
    # Build and train Logistic Regression model
    print("Building and training Logistic Regression model...")
    logreg_model = build_logistic_regression_model()
    logreg_model = train_logistic_regression_model(logreg_model, X_train, y_train_scaled)
    
    # Evaluate Logistic Regression model
    print("Evaluating Logistic Regression model...")
    logreg_results = evaluate_logistic_regression_model(logreg_model, X_test, y_test_scaled)
    
    # Define run name
    model_type = "LSTM-BERT" if use_bert else "LSTM"
    run_name = f"{model_type}_emb{embedding_dim}_vocab{trunc_vocab_size}"
    
    # After training
    with mlflow.start_run(run_name=run_name):
        log_model_training(
            model=lstm_model,
            history=lstm_history,
            metrics=lstm_metrics,
            params=params
        )


    # Save all models and artifacts
    save_models(
        lstm_model=lstm_model,
        logreg_model=logreg_model,
        tokenizer=tokenizer,
        target_scaler=target_scaler
    )
    
    return {
        'lstm_model': lstm_model,
        'logreg_model': logreg_model,
        'tokenizer': tokenizer,
        'target_scaler': target_scaler,
        'lstm_metrics': lstm_metrics,
        'logreg_results': logreg_results
    }

def main():
    """
    Main function to run when executing the script directly.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='News Sentiment Analysis Tool')
    parser.add_argument('--train', action='store_true', help='Train new models from scratch')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data file')
    parser.add_argument('--analyze', type=str, default=None, help='Path to news texts file for analysis')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.train:
        if args.data is None:
            print("Error: Please provide a data file path with --data")
            return
        
        print(f"Loading data from {args.data}")
        df = load_dataset(args.data)
        
        print("Training new models...")
        results = train_new_models(df, save_dir='saved_models')
        
        print("Training completed!")
        print(f"LSTM model performance: {results['lstm_metrics']}")
        
    elif args.analyze:
        # Load the file containing news to analyze
        if os.path.exists(args.analyze):
            with open(args.analyze, 'r', encoding='utf-8') as f:
                news_texts = f.readlines()
            
            # Remove any empty lines
            news_texts = [text.strip() for text in news_texts if text.strip()]
            
            if not news_texts:
                print("Error: No news texts found in the file")
                return
            
            print(f"Analyzing {len(news_texts)} news texts...")
            
            # Process and analyze the news
            results = process_and_analyze_news(
                news_texts, 
                use_existing_models=True,
                lstm_model_path='saved_models/lstm_bert_model.h5',
                lr_model_path='saved_models/logreg_model.pkl'
            )
            
            # Export results
            if results['ranked_news'] is not None:
                export_path = os.path.join(args.output, 'ranked_news.csv')
                export_ranked_news(results['ranked_news'], export_path)
                print(f"Results exported to {export_path}")
                
                # Display top 5 most important news
                print("\nTop 5 most financially important news:")
                for i, (idx, row) in enumerate(results['top_news'].head(5).iterrows(), 1):
                    print(f"{i}. {row['news_text'][:100]}... (Importance: {row['financial_importance']:.4f})")
            else:
                print("Could not generate rankings due to model loading issues.")
        else:
            print(f"Error: File not found - {args.analyze}")
    
    else:
        print("Please specify either --train or --analyze mode.")
        print("Example usage:")
        print("  To train: python main.py --train --data=/path/to/news.csv")
        print("  To analyze: python main.py --analyze=/path/to/news_to_analyze.txt --output=results")

if __name__ == "__main__":
    main()
