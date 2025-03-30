import os
import mlflow
import dagshub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import os
import mlflow
import dagshub
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_mlflow(experiment_name="news_sentiment_analysis", use_local=False):
    """
    Configure MLflow tracking URI and authenticate with DagsHub
    
    Parameters:
    -----------
    experiment_name: str
        Name of the experiment
    use_local: bool
        Whether to use local tracking instead of DagsHub
        
    Returns:
    --------
    mlflow instance
    """
    if use_local:
        print("Using local MLflow tracking")
        # Set local tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
    else:
        # Try to use configured tracking
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
        else:
            print("Warning: MLFLOW_TRACKING_URI not found in .env file")
            print("Using default local tracking URI")
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Fix: Updated DagsHub authentication method
        try:
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            dagshub_user = os.getenv("DAGSHUB_USERNAME")
            repo_name = os.getenv("DAGSHUB_REPO_NAME", "news-sentiment-analysis")
            
            if dagshub_token and dagshub_user:
                # The correct parameter format for dagshub.init()
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                
                # Initialize connection without token parameter
                dagshub.init(repo_owner=dagshub_user, repo_name=repo_name)
                print(f"Successfully connected to DagsHub repository: {dagshub_user}/{repo_name}")
            else:
                print("Warning: DagsHub credentials not found, using local tracking only")
        except Exception as e:
            print(f"Warning: Failed to connect to DagsHub: {str(e)}")
            print("Continuing with local tracking")
    
    # Safely set/create experiment using try/except
    try:
        # Set existing experiment if it exists
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Error creating experiment: {str(e)}")
        print("Using default experiment")
        # Fall back to default experiment
        
    return mlflow

def log_model_training(model, history, hyperparams, metrics=None, model_name="sentiment_model"):
    """
    Log model training results to MLflow
    
    Parameters:
    -----------
    model: Keras model
        The trained model
    history: History object
        Training history from model.fit()
    hyperparams: dict
        Model hyperparameters
    metrics: dict
        Additional metrics to log
    model_name: str
        Name to save the model under
    """
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Log metrics from history
        for epoch in range(len(history.history['loss'])):
            for metric_name, values in history.history.items():
                mlflow.log_metric(metric_name, values[epoch], step=epoch)
        
        # Log additional evaluation metrics
        if metrics:
            mlflow.log_metrics(metrics)
            
        # Log model artifacts
        mlflow.keras.log_model(model, model_name)
        
        # Log model summary as text artifact
        model_summary_path = "model_summary.txt"
        with open(model_summary_path, 'w') as f:
            # Redirect model summary to file
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact(model_summary_path)
        os.remove(model_summary_path)  # Clean up

def log_prediction_results(y_true, y_pred, class_names=None):
    """
    Log prediction metrics and confusion matrix
    
    Parameters:
    -----------
    y_true: array-like
        True labels
    y_pred: array-like
        Predicted labels
    class_names: list
        Names of classes
    """
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True, target_names=class_names)
    
    # Log classification metrics
    for class_name in report:
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            for metric_name, value in report[class_name].items():
                if metric_name != 'support':
                    mlflow.log_metric(f"{class_name}_{metric_name}", value)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save and log the plot
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)  # Clean up