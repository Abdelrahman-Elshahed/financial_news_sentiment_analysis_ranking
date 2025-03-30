import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from src.experiment_tracking import setup_mlflow

def generate_sample_data():
    """Generate balanced sample news data with clear sentiment signals"""
    print("Creating balanced sample data for testing...")
    
    # Create clear positive examples
    pos_news = [
        "Company reports strong earnings, stock price jumps 15%",
        "Stock market reaches all-time high as investor confidence soars",
        "Tech company announces revolutionary new product, shares surge",
        "Economic growth beats expectations by wide margin, outlook positive",
        "Company successfully acquires competitor, creating market leader"
    ]
    
    # Create clear negative examples
    neg_news = [
        "Market crashes amid severe recession fears, worst day in years",
        "Company announces massive layoffs following disastrous earnings", 
        "Tech giant faces regulatory investigation, stock plummets",
        "Oil prices collapse as demand forecasts deteriorate",
        "CEO resigns amid accounting scandal, shares in freefall"
    ]
    
    # CORRECT scores to ensure different classes:
    # Positive news has high positive & low negative scores
    pos_scores = np.array([0.9, 0.85, 0.8, 0.95, 0.75])
    pos_neg_scores = 1 - pos_scores  # Low negative scores for positive news
    
    # Negative news has high negative & low positive scores
    neg_scores = np.array([0.9, 0.85, 0.8, 0.95, 0.75])
    neg_pos_scores = 1 - neg_scores  # Low positive scores for negative news
    
    # Combine the data CORRECTLY
    news_texts = pos_news + neg_news
    all_pos_scores = np.concatenate([pos_scores, neg_pos_scores])  # Fixed: Use low pos for neg news
    all_neg_scores = np.concatenate([pos_neg_scores, neg_scores])  # Fixed: Use high neg for neg news
    
    # Create DataFrame
    df = pd.DataFrame({
        'news': news_texts,
        'pos': all_pos_scores,
        'neg': all_neg_scores
    })
    
    # Save for future use
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/financial_news.csv", index=False)
    print(f"Created balanced sample data with {len(df)} examples")
    
    return df


def preprocess_data(texts):
    """Basic preprocessing for text data"""
    # Create a tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
    
    return padded, tokenizer

def build_model(vocab_size=10000, embedding_dim=100):
    """Build a simple sentiment analysis model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=100),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    return model

def train_and_log(X_train, y_train, X_val, y_val, params):
    """Train model and log with MLflow"""
    # Set up MLflow
    mlflow = setup_mlflow(use_local=False, experiment_name="lstm_experiments")


    # Add model type to parameters
    if "model_type" not in params:
        params["model_type"] = "BiLSTM"  # Explicitly record model architecture    
    # Build model based on parameters
    model = build_model(
        vocab_size=params["vocab_size"],
        embedding_dim=params["embedding_dim"]
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print(f"Training model with params: {params}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=1
    )
    
    # Get model type and create descriptive run name
    model_type = params.get("model_type", "BiLSTM")
    run_name = f"{model_type}_units{params['lstm_units']}_dropout{params['dropout_rate']}"
    
    # Log with MLflow using descriptive run name
    with mlflow.start_run(run_name=run_name):
        # Set a tag for model type
        mlflow.set_tag("model_type", model_type)

        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        for epoch, metrics in enumerate(zip(
            history.history["loss"],
            history.history["val_loss"],
            history.history["accuracy"],
            history.history["val_accuracy"]
        )):
            loss, val_loss, acc, val_acc = metrics
            mlflow.log_metrics({
                "loss": loss,
                "val_loss": val_loss,
                "accuracy": acc,
                "val_accuracy": val_acc
            }, step=epoch)
        
        # Log model
        mlflow.keras.log_model(model, "model")
    
    return model, history

def run_experiment():
    """Run sentiment analysis experiment with MLflow tracking"""
    # Create or load data
    try:
        df = pd.read_csv("data/financial_news.csv")
        print("Loaded existing dataset")
    except FileNotFoundError:
        df = generate_sample_data()
    
    # Add debug prints
    print("\nDataframe sample:")
    print(df.head(2))
    print(f"\nTarget distribution:\nPositive: {df['pos'].mean():.2f} (avg)\nNegative: {df['neg'].mean():.2f} (avg)")
    
    # Convert to proper binary classification format for LogisticRegression
    # For each news item, determine the dominant sentiment (pos or neg)
    df['dominant_sentiment'] = (df['pos'] > df['neg']).astype(int)
    print(f"\nDominant sentiment counts: {df['dominant_sentiment'].value_counts().to_dict()}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # For LSTM: Use continuous scores
    X = df["news"].values
    y_continuous = np.column_stack((df["pos"].values, df["neg"].values))
    
    # For LogReg: Use binary class
    y_binary = df["dominant_sentiment"].values
    
    # Split with stratification on dominant sentiment
    X_train, X_test, y_train_cont, y_test_cont, y_train_bin, y_test_bin = train_test_split(
        X, y_continuous, y_binary, test_size=0.2, random_state=42, 
        stratify=df["dominant_sentiment"]  # Stratify to preserve class balance
    )
    
    X_train, X_val, y_train_cont, y_val_cont, y_train_bin, y_val_bin = train_test_split(
        X_train, y_train_cont, y_train_bin, test_size=0.25, random_state=42,
        stratify=y_train_bin  # Stratify to preserve class balance
    )
    
    # Print class distributions to verify
    print(f"\nBinary class distribution:")
    print(f"  Training: {np.bincount(y_train_bin)}")
    print(f"  Validation: {np.bincount(y_val_bin)}")
    print(f"  Test: {np.bincount(y_test_bin)}")
    
    # Preprocess text data for LSTM
    X_train_processed, tokenizer = preprocess_data(X_train)
    X_val_processed = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_val), maxlen=100
    )
    X_test_processed = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=100
    )
    
    print("\n--- Training LSTM model ---")
    # Train and track LSTM model with continuous scores
    lstm_model, history = train_and_log(
        X_train_processed, y_train_cont,
        X_val_processed, y_val_cont,
        {
            "model_type": "BiLSTM",
            "vocab_size": 10000,
            "embedding_dim": 100,
            "lstm_units": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 2,
            "epochs": 10
        }
    )
    
    print("\n--- Training Logistic Regression model ---")
    # Train and track logistic regression model with binary targets
    from src.train import train_logistic_regression_with_tracking
    
    # Fix: Create a modified train_logistic_regression function for binary classification
    def train_binary_logistic_regression(X_train, y_train, X_val, y_val, X_test=None, y_test=None, **kwargs):
        """
        Train binary logistic regression model (not multioutput)
        """
        # Setup MLflow
        from src.experiment_tracking import setup_mlflow
        mlflow = setup_mlflow(use_local=False, experiment_name="logistic_regression_experiments")
        
        # Extract parameters
        params = {
            "max_features": kwargs.get("max_features", 5000),
            "C": kwargs.get("C", 1.0),
            "max_iter": kwargs.get("max_iter", 1000),
            "solver": kwargs.get("solver", "liblinear"),
            "ngram_range": kwargs.get("ngram_range", (1, 1)),
            "use_idf": kwargs.get("use_idf", True),
        }
        
        # Build simple logistic regression for binary classification
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        # Single output LogisticRegression (not MultiOutputRegressor)
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=params["max_features"],
                ngram_range=params["ngram_range"],
                use_idf=params["use_idf"]
            )),
            ('lr', LogisticRegression(
                C=params["C"],
                max_iter=params["max_iter"],
                solver=params["solver"],
                random_state=42
            ))
        ])
        
        # Print class distribution
        print(f"Training class distribution: {np.bincount(y_train)}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Validation predictions
        val_pred_proba = model.predict_proba(X_val)[:, 1]  # Get probability of positive class
        val_pred = model.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)
        val_roc_auc = roc_auc_score(y_val, val_pred_proba)
        
        # Create run name
        run_name = f"LogReg_binary_features{params['max_features']}_C{params['C']}"
        
        # Log with MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", "LogisticRegression_Binary")
            mlflow.log_params(params)
            
            # Log validation metrics
            mlflow.log_metrics({
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "val_roc_auc": val_roc_auc
            })
            
            # Log test metrics if available
            if X_test is not None and y_test is not None:
                test_pred = model.predict(X_test)
                test_pred_proba = model.predict_proba(X_test)[:, 1]
                
                test_accuracy = accuracy_score(y_test, test_pred)
                test_f1 = f1_score(y_test, test_pred)
                test_roc_auc = roc_auc_score(y_test, test_pred_proba)
                
                mlflow.log_metrics({
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "test_roc_auc": test_roc_auc
                })
            
            # Log model
            mlflow.sklearn.log_model(model, "logistic_regression_model")
            
            # Log top features
            try:
                feature_names = model.named_steps['tfidf'].get_feature_names_out()
                coefficients = model.named_steps['lr'].coef_[0]
                
                # Create feature importance dataframe
                import pandas as pd
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coefficients
                })
                
                # Sort by absolute importance
                feature_importance['abs_importance'] = abs(feature_importance['importance'])
                feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
                
                # Save top features
                top_features = feature_importance.head(20)
                top_features_path = "top_features.csv"
                top_features.to_csv(top_features_path, index=False)
                
                # Log artifact
                mlflow.log_artifact(top_features_path)
                
                # Clean up
                import os
                os.remove(top_features_path)
            except Exception as e:
                print(f"Could not log feature importances: {e}")
        
        return model
    
    # Call the binary version with correct targets
    logreg_model = train_binary_logistic_regression(
        X_train, y_train_bin,
        X_val, y_val_bin,
        X_test, y_test_bin,
        max_features=1000,
        C=1.0,
        ngram_range=(1, 2)
    )
    
    print("Experiment completed!")

if __name__ == "__main__":
    run_experiment()