# Financial News Sentiment Analysis & Ranking

This project implements a comprehensive solution for analyzing sentiment in financial news and ranking articles based on their financial importance. It uses deep learning models (LSTM with BERT embeddings) and traditional machine learning approaches (Logistic Regression) to predict sentiment scores for news articles.



## Table of Contents

  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Features](#features)
  - [Primary Model Architecture](#primary-model-architecture)
  - [Setup](#setup)
  - [Test Suite](#test-suite)
  - [Run with Streamlit Application](#run-with-streamlit-application)
  - [API Usage and PostmanAPI Testing](#api-usage-and-postmanapi-testing)
  - [Dockerization](#dockerization)
  - [MLflow Integration and DagsHub](#mlflow-integration-and-dagshub)


## Overview

This project analyzes financial news articles to determine sentiment (positive, negative, neutral) and ranks companies based on their news sentiment scores. By processing real-time news data, the system provides valuable insights for investors and financial analysts to make informed decisions.




## Project Structure
```bash
financial_news_sentiment_analysis_ranking/
├── assets/                # Screenshots and images for documentation
├── data/                  # Input data files (financial_news.csv, news.csv)
├── deployment/            # Deployment-related code
│   ├── app.py             # FastAPI application for serving predictions
│   ├── run.py             # Script to run both API and Streamlit services
│   └── streamlit_app.py   # Streamlit web interface
├── models/                # Saved models and tokenizers
├── notebooks/             # Jupyter notebooks
├── src/                   # Core source code
│   ├── data_loader.py     # Functions to load and prepare datasets
│   ├── experiment_tracking.py # MLflow integration
│   ├── main.py            # Main entry point and orchestration
│   ├── model.py           # Model architectures definition
│   ├── preprocessing.py   # Text preprocessing utilities
│   ├── ranking.py         # News ranking functionality
│   ├── run_experiment.py  # MLflow experiment runner
│   ├── sentiment_analysis.py # Sentiment prediction and analysis
│   └── train.py           # Model training functions
└── tests/                 # Unit tests
   ```



## Features

- Text Preprocessing: Lowercasing, punctuation removal, stopword removal
- Multiple Model Types: LSTM with BERT embeddings and Logistic Regression
- Sentiment Analysis: Predict positive, negative, and neutral sentiment scores
- News Ranking: Rank financial news by importance
- Model Persistence: Save and load trained models and tokenizers
- Experiment Tracking: Track experiments with MLflow
- API Deployment: Serve predictions through FastAPI
- Web Interface: User-friendly interface with Streamlit
- Containerization: Docker support for easy deployment



## Primary Model Architecture

The primary model is an LSTM network with BERT embeddings that processes financial news text and outputs sentiment scores. The architecture includes:

- Text tokenization and padding
- Embedding layer (using BERT embeddings)
- Bidirectional LSTM layers
- Dropout for regularization
- Dense output layer for sentiment prediction





## Setup

- Clone the Repository

   ```bash
   git clone https://github.com/Abdelrahman-Elshahed/financial_news_sentiment_analysis_ranking.git
   ```
- Create and activate a virtual environment:
  ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
  ```
- Set up dependencies

  Install Python dependencies:
  To install all the required Python packages for the project, run the following command:
  ```bash
  pip install -r requirements.txt
  ```




## Test Suite 


This directory contains tests for the Financial News Sentiment Analysis & Ranking project.

  ### Test Files
- conftest.py: Shared pytest fixtures (sample data, mock model, mock tokenizer)
- test_data_loader.py: Tests for data loading and preparation functions
- test_preprocessing.py: Tests for text preprocessing utilities
- test_sentiment_analysis.py: Tests for sentiment prediction functionality


![Image](https://github.com/user-attachments/assets/376a8e03-7c93-4c6c-a1de-5e269e8176eb)




## Run with Streamlit Application

   - Run the combined service (API + Streamlit)
     ```bash
     cd deployment
     python run.py
     ```
  - Or run them separately
       ```bash
    cd deployment
    uvicorn app:app --host 0.0.0.0 --port 8000
    streamlit run streamlit_app.py
     ```

![Image](https://github.com/user-attachments/assets/becabe10-2360-41d6-ab03-44d69075fb1d)

![Image](https://github.com/user-attachments/assets/79e079b8-0883-40e0-85a9-33ced0c259e9)

![Image](https://github.com/user-attachments/assets/7567ff61-a648-4b3e-93e0-8ec11984a7a1)


## API Usage and PostmanAPI Testing

- GET /: Health check
- POST /predict: Predict sentiment for a single news article
- POST /predict_batch: Predict sentiment for multiple news articles

![Image](https://github.com/user-attachments/assets/54a48005-6667-41ad-9573-d1b1c9a724c2)

![Image](https://github.com/user-attachments/assets/4ff47b44-c27b-40fd-8741-215a84ea02e5)

![Image](https://github.com/user-attachments/assets/29558702-701a-458e-b043-e0a991d41952)




## Dockerization

   - Build the Docker image with:
     ```bash
     docker build -t financial-news-sentiment .
     ```
   - Run the container with:
     ```bash
     docker run -p 8000:8000 -p 8501:8501 financial-news-sentiment
     ```
  ![Image](https://github.com/user-attachments/assets/9f83c872-97ee-4927-920f-f92ecd06f65c)

  
  ### Docker image on Docker Hub [Click Here](https://hub.docker.com/repository/docker/bodaaa/financial-news-sentiment/general).
  ## 🔧 Usage

Pull the image:

```bash
docker pull bodaaa/financial-news-sentiment:latest
```
Run the container:


```
docker run -p 8000:8000 -p 8501:8501 bodaaa/financial-news-sentiment:latest
```

  ![Image](https://github.com/user-attachments/assets/d9be4a43-ca71-4fb6-81a0-d7b85c3cd664)

## MLflow Integration and DagsHub

The project uses MLflow to track experiments, including:

- Model parameters
- Training and validation metrics
- Saved model artifacts
  
Access the MLflow UI to compare different model configurations and results.
![Image](https://github.com/user-attachments/assets/5085c2fb-0cc7-4915-8423-c7d8684cd0e4)


### DagsHub Integration
- For DagsHub Experiments [Click Here](https://dagshub.com/Abdelrahman-Elshahed/news-sentiment-analysis/experiments).

![Image](https://github.com/user-attachments/assets/f6d7d2ee-4ea9-48bf-8689-f2829e33f73a)

![Image](https://github.com/user-attachments/assets/71bb51e6-74f7-43bf-80b1-17490f3e3b47)

![Image](https://github.com/user-attachments/assets/ed4768c8-0ab6-486d-a429-f5f192cd49b1)
