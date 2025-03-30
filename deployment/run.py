import subprocess
import os
import time
import sys

# Determine if we're running from deployment dir or project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DEPLOYMENT_DIR = os.path.basename(SCRIPT_DIR) == "deployment"

def start_fastapi():
    """Start the FastAPI backend server"""
    print("Starting FastAPI backend...")
    
    # Configure paths based on where we're running from
    if IN_DEPLOYMENT_DIR:
        # Already in deployment directory
        cmd = ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    else:
        # In project root
        cmd = ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
    
    subprocess.Popen(cmd)    
    time.sleep(3)  # Give the server time to start

def start_streamlit():
    """Start the Streamlit frontend"""
    print("Starting Streamlit frontend...")
    
    # Configure paths based on where we're running from
    if IN_DEPLOYMENT_DIR:
        # Already in deployment directory
        cmd = ["streamlit", "run", "streamlit_app.py", 
               "--server.port=8501", "--server.address=0.0.0.0"]
    else:
        # In project root
        cmd = ["streamlit", "run", "deployment/streamlit_app.py", 
               "--server.port=8501", "--server.address=0.0.0.0"]
    
    subprocess.Popen(cmd)

def main():
    # Check if model files exist with correct paths
    if IN_DEPLOYMENT_DIR:
        # Running from deployment dir
        model_path = "../models/lstm_bert_model.h5"
        tokenizer_path = "../models/tokenizer.pkl"
    else:
        # Running from project root
        model_path = "models/lstm_bert_model.h5"
        tokenizer_path = "models/tokenizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("WARNING: Model files not found. The application might not work correctly.")
        print(f"Expected files: {model_path} and {tokenizer_path}")
    else:
        print("Model files found successfully!")
        
    print("Starting services...")
    start_fastapi()
    start_streamlit()
    
    # Keep the main process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main()