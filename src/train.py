import os
import pickle
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, precision_recall_curve, auc
from src.experiment_tracking import setup_mlflow, log_model_training, log_prediction_results
import tensorflow as tf
from src.model import build_sentiment_model

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    
    # Visualize training history if available
    try:
        from sentiment_analysis import plot_training_history
        plot_training_history(history)
    except ImportError:
        pass
    
    return history


def evaluate_lstm_model(model, X_test, y_test):
    
    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Visualize confusion matrices if available
    try:
        from sentiment_analysis import plot_confusion_matrices
        plot_confusion_matrices(y_test, y_pred)
    except ImportError:
        pass
    
    return {
        'loss': loss,
        'mae': mae,
        'predictions': y_pred
    }

def evaluate_binary_lstm_model(model, X_test, y_test):

    # Convert one-hot encoded labels to class indices if needed
    if len(y_test.shape) > 1 and y_test.shape[1] == 2:
        y_test_classes = np.argmax(y_test, axis=1)
    else:
        y_test_classes = y_test
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    f1 = f1_score(y_test_classes, y_pred_classes)
    report = classification_report(y_test_classes, y_pred_classes, output_dict=True)
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test_classes, y_pred_proba[:, 1])
    except:
        roc_auc = None
    
    # Plot confusion matrix if available
    try:
        from sentiment_analysis import plot_confusion_matrices
        from tensorflow.keras.utils import to_categorical
        plot_confusion_matrices(
            to_categorical(y_test_classes, 2),  # Convert to one-hot for the plot function
            y_pred_proba,
            labels=['Positive', 'Negative']
        )
    except ImportError:
        pass
        
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'roc_auc': roc_auc,
        'predictions': y_pred_proba
    }

def convert_to_binary(true_values, pred_values, threshold=0.5):
    
    return (true_values > threshold).astype(int), (pred_values > threshold).astype(int)

def evaluate_classification_metrics(y_true, y_pred, sentiment_labels=None):
    
    if sentiment_labels is None:
        sentiment_labels = ['Positive', 'Negative', 'Neutral']
    
    metrics = {}
    
    for i, sentiment in enumerate(sentiment_labels):
        # Convert continuous values to binary classes
        y_true_binary, y_pred_binary = convert_to_binary(y_true[:, i], y_pred[:, i])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        report = classification_report(y_true_binary, y_pred_binary, output_dict=True)
        
        # Calculate threshold-independent metrics
        try:
            roc_auc = roc_auc_score(y_true_binary, y_pred[:, i])
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred[:, i])
            pr_auc = auc(recall, precision)
        except:
            roc_auc = None
            pr_auc = None
        
        metrics[sentiment] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'report': report,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    return metrics

def train_logistic_regression_model(model, X_train, y_train):
    
    # Convert targets to binary (0 or 1) for classification
    y_train_binary = (y_train > 0.5).astype(int)
    
    # Train the model
    model.fit(X_train, y_train_binary)
    
    return model

def evaluate_logistic_regression_model(model, X_test, y_test):

    # Convert targets to binary (0 or 1) for classification
    y_test_binary = (y_test > 0.5).astype(int)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    # Generate classification report
    report = classification_report(y_test_binary, y_pred_binary, output_dict=True)
    
    # Calculate ROC AUC when possible
    try:
        roc_aucs = []
        for i in range(y_test_binary.shape[1]):
            roc_aucs.append(roc_auc_score(y_test_binary[:, i], y_pred[:, i]))
        roc_auc = np.mean(roc_aucs)
    except:
        roc_auc = None
    
    # Visualize confusion matrix if available
    try:
        from sentiment_analysis import plot_confusion_matrices
        plot_confusion_matrices(y_test_binary, y_pred_binary)
    except ImportError:
        pass
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'roc_auc': roc_auc,
        'predictions': y_pred
    }

def save_models(lstm_model=None, logreg_model=None, tokenizer=None, target_scaler=None):

    os.makedirs('saved_models', exist_ok=True)
    
    if lstm_model:
        lstm_model.save('saved_models/lstm_bert_model.h5')
        print("LSTM model saved to saved_models/lstm_bert_model.h5")
        
    if logreg_model:
        with open('saved_models/logreg_model.pkl', 'wb') as f:
            pickle.dump(logreg_model, f)
        print("Logistic Regression model saved to saved_models/logreg_model.pkl")
        
    if tokenizer:
        with open('saved_models/tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        print("Tokenizer saved to saved_models/tokenizer.pkl")
        
    if target_scaler:
        with open('saved_models/target_scaler.pkl', 'wb') as f:
            pickle.dump(target_scaler, f)
        print("Target scaler saved to saved_models/target_scaler.pkl")

from src.experiment_tracking import setup_mlflow, log_model_training, log_prediction_results

def train_sentiment_model(X_train, y_train, X_val, y_val, X_test=None, y_test=None, **kwargs):
    """
    Train a sentiment analysis model and track with MLflow
    
    Parameters:
    -----------
    X_train: numpy.ndarray
        Training features
    y_train: numpy.ndarray
        Training labels
    X_val: numpy.ndarray
        Validation features
    y_val: numpy.ndarray
        Validation labels
    X_test: numpy.ndarray, optional
        Test features
    y_test: numpy.ndarray, optional
        Test labels
    **kwargs: 
        Additional parameters for model configuration
        
    Returns:
    --------
    model: tf.keras.Model
        Trained model
    history: tf.keras.callbacks.History
        Training history
    """
    # Setup MLflow
    from src.experiment_tracking import setup_mlflow
    mlflow = setup_mlflow()
    
    # Extract hyperparameters with defaults
    params = {
        "embedding_dim": kwargs.get("embedding_dim", 100),
        "max_length": kwargs.get("max_length", 100),
        "vocab_size": kwargs.get("vocab_size", 10000),
        "lstm_units": kwargs.get("lstm_units", 128),
        "dropout_rate": kwargs.get("dropout_rate", 0.2),
        "learning_rate": kwargs.get("learning_rate", 0.001),
        "batch_size": kwargs.get("batch_size", 32),
        "epochs": kwargs.get("epochs", 10),
    }
    
    # Build model
    model = build_sentiment_model(
        vocab_size=params["vocab_size"],
        embedding_dim=params["embedding_dim"],
        max_length=params["max_length"],
        lstm_units=params["lstm_units"],
        dropout_rate=params["dropout_rate"]
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks
    )
    
    # Create descriptive run name
    model_type = "BiLSTM"  # You can make this a parameter if needed
    run_name = f"{model_type}_units{params['lstm_units']}_dropout{params['dropout_rate']}"
    
    # Log training metrics with descriptive run name
    with mlflow.start_run(run_name=run_name):
        # Set a tag for model type
        mlflow.set_tag("model_type", model_type)

        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics from training
        for epoch, (loss, val_loss, acc, val_acc) in enumerate(zip(
            history.history['loss'],
            history.history['val_loss'],
            history.history['accuracy'],
            history.history['val_accuracy']
        )):
            mlflow.log_metrics({
                "loss": loss,
                "val_loss": val_loss,
                "accuracy": acc,
                "val_accuracy": val_acc
            }, step=epoch)
        
        # Test if test data is provided
        if X_test is not None and y_test is not None:
            test_results = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({
                "test_loss": test_results[0],
                "test_accuracy": test_results[1]
            })
        
        # Log model
        mlflow.keras.log_model(model, "model")
    
    return model, history

def train_logistic_regression_with_tracking(X_train, y_train, X_val, y_val, X_test=None, y_test=None, **kwargs):
    """
    Train a logistic regression model with MLflow tracking
    
    Parameters:
    -----------
    X_train: array-like
        Training features
    y_train: array-like
        Training labels
    X_val: array-like
        Validation features
    y_val: array-like
        Validation labels
    X_test: array-like, optional
        Test features
    y_test: array-like, optional
        Test labels
    **kwargs: dict
        Additional parameters
        
    Returns:
    --------
    model: sklearn Pipeline
        Trained logistic regression model
    """
    # Setup MLflow
    from src.experiment_tracking import setup_mlflow
    mlflow = setup_mlflow()
    
    # Extract hyperparameters with defaults
    params = {
        "max_features": kwargs.get("max_features", 5000),
        "C": kwargs.get("C", 1.0),
        "max_iter": kwargs.get("max_iter", 1000),
        "solver": kwargs.get("solver", "liblinear"),
        "ngram_range": kwargs.get("ngram_range", (1, 1)),
        "use_idf": kwargs.get("use_idf", True),
    }
    
    # Build logistic regression model with TF-IDF
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import LogisticRegression
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            use_idf=params["use_idf"]
        )),
        ('lr', MultiOutputRegressor(LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"]
        )))
    ])
    
    # Convert targets to binary (0 or 1) for classification if needed
    y_train_binary = (y_train > 0.5).astype(int) if hasattr(y_train, 'astype') else y_train
    y_val_binary = (y_val > 0.5).astype(int) if hasattr(y_val, 'astype') else y_val
    
    # Train the model
    model.fit(X_train, y_train_binary)
    
    # Create descriptive run name
    model_type = "LogisticRegression"
    run_name = f"{model_type}_features{params['max_features']}_C{params['C']}"
    
    # Get predictions for validation and test sets
    val_pred = model.predict(X_val)
    val_pred_binary = (val_pred > 0.5).astype(int)

    def check_class_distribution(y, name="dataset"):
        if len(np.unique(y)) < 2:
            print(f"WARNING: {name} contains only one class ({np.unique(y)[0]})")
            return False
        class_counts = np.bincount(y.astype(int))
        print(f"{name} class distribution: {class_counts}")
        return len(class_counts) > 1

    # Then use it before calculating metrics:
    for i in range(y_val_binary.shape[1]):
        if check_class_distribution(y_val_binary[:, i], f"Validation data (class {i})"):
            val_metrics[f"val_f1_class{i}"] = f1_score(y_val_binary[:, i], val_pred_binary[:, i])
            val_metrics[f"val_roc_auc_class{i}"] = roc_auc_score(y_val_binary[:, i], val_pred[:, i])
        else:
            # Log zeros with a tag explaining why
            val_metrics[f"val_f1_class{i}"] = 0
            val_metrics[f"val_roc_auc_class{i}"] = 0
            mlflow.set_tag(f"class{i}_metrics_warning", "Only one class in validation data")
            
    # Calculate validation metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    val_accuracy = accuracy_score(y_val_binary, val_pred_binary)
    
    # Calculate F1 and ROC scores per class
    val_metrics = {}
    
    for i in range(y_val_binary.shape[1]):
        try:
            val_metrics[f"val_f1_class{i}"] = f1_score(y_val_binary[:, i], val_pred_binary[:, i])
            val_metrics[f"val_roc_auc_class{i}"] = roc_auc_score(y_val_binary[:, i], val_pred[:, i])
        except Exception as e:
            print(f"Error calculating metrics for class {i}: {e}")
    
    # Log with MLflow
    with mlflow.start_run(run_name=run_name):
        # Set a tag for model type
        mlflow.set_tag("model_type", model_type)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log validation metrics
        mlflow.log_metric("val_accuracy", val_accuracy)
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Log test metrics if test data is provided
        if X_test is not None and y_test is not None:
            y_test_binary = (y_test > 0.5).astype(int) if hasattr(y_test, 'astype') else y_test
            test_pred = model.predict(X_test)
            test_pred_binary = (test_pred > 0.5).astype(int)
            
            test_accuracy = accuracy_score(y_test_binary, test_pred_binary)
            mlflow.log_metric("test_accuracy", test_accuracy)
            
            # Log test metrics per class
            for i in range(y_test_binary.shape[1]):
                try:
                    mlflow.log_metric(
                        f"test_f1_class{i}", 
                        f1_score(y_test_binary[:, i], test_pred_binary[:, i])
                    )
                    mlflow.log_metric(
                        f"test_roc_auc_class{i}", 
                        roc_auc_score(y_test_binary[:, i], test_pred[:, i])
                    )
                except Exception as e:
                    print(f"Error calculating test metrics for class {i}: {e}")
        
        # Log the model
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        
        # Log feature importances if possible
        try:
            # Get feature names from TF-IDF
            feature_names = model.named_steps['tfidf'].get_feature_names_out()
            
            # Get coefficients for each output
            coefs = model.named_steps['lr'].estimators_
            
            # For each output class
            for i, estimator in enumerate(coefs):
                # Get coefficients
                coefficients = estimator.coef_[0]
                
                # Create dataframe of feature importances
                import pandas as pd
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coefficients
                })
                
                # Sort by absolute importance
                feature_importance['abs_importance'] = abs(feature_importance['importance'])
                feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
                
                # Save top features to CSV
                top_features = feature_importance.head(100)
                top_features_path = f"class{i}_top_features.csv"
                top_features.to_csv(top_features_path, index=False)
                
                # Log as artifact
                mlflow.log_artifact(top_features_path)
                
                # Clean up
                os.remove(top_features_path)
                
        except Exception as e:
            print(f"Could not log feature importances: {e}")
    
    return model