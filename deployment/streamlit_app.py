import streamlit as st
import pandas as pd
import requests

# Set page config
st.set_page_config(
    page_title="News Sentiment Analysis",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint URL - change this when deploying
API_URL = "http://localhost:8000"  # Default for local testing

# Sidebar for API configuration
with st.sidebar:
    st.title("📊 Configuration")
    custom_api_url = st.text_input("API URL", value=API_URL)
    if custom_api_url:
        API_URL = custom_api_url
    
    st.write(f"Using API at: {API_URL}")
    
    # Test API connection
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{API_URL}/")
            if response.status_code == 200:
                st.success("✅ API is online!")
            else:
                st.error(f"❌ API returned status code {response.status_code}")
        except Exception as e:
            st.error(f"❌ Failed to connect to API: {e}")

# Title and description
st.title("📰 Financial News Sentiment Analysis")
st.markdown("""
This application analyzes the sentiment of financial news articles using a deep learning model.
Enter financial news texts to predict whether the sentiment is positive or negative.
""")

# Tab system for different functionality
tab1, tab2 = st.tabs(["News Analysis", "About"])

# News Analysis Tab
with tab1:
    st.header("Analyze News Article")
    
    news_text = st.text_area(
        "Enter news text", 
        height=150,
        placeholder="Example: The company reported strong earnings and a record high stock price."
    )
    
    if st.button("Analyze Sentiment", key="analyze_single"):
        if news_text:
            with st.spinner("Analyzing sentiment..."):
                try:
                    response = requests.post(
                        f"{API_URL}/analyze", 
                        json={"text": news_text}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Get sentiment and score directly from the API response
                        sentiment = result["sentiment"]
                        sentiment_score = result["sentiment_score"]  # Already adjusted by +0.5 in the API
                        
                        # Create markdown content with bullet points
                        markdown_result = f"""
                        - Input Text: {news_text}
                        - Predicted Sentiment: {sentiment}
                        - Sentiment Score: {sentiment_score:.4f}
                        - Rank Score: {result['rank_score']:.4f}
                        """
                        
                        # Display with appropriate styling based on sentiment
                        if sentiment == "POSITIVE":
                            st.success("Analysis Result")
                        else:
                            st.error("Analysis Result")
                            
                        # Display the bulleted result
                        st.markdown(markdown_result)
                            
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to analyze sentiment: {e}")
        else:
            st.warning("Please enter some news text to analyze.")

# About Tab
with tab2:
    st.header("About This Application")
    
    st.markdown("""
    ### News Sentiment Analysis Tool

    This application uses a deep learning model to analyze the sentiment of financial news articles.
    The model has been trained on financial news data to identify positive or negative sentiment.

    #### How It Works

    1. **Text Preprocessing**: The application cleans and standardizes news text by removing special characters,
       normalizing text, and filtering out common stop words.
    
    2. **Sentiment Analysis**: A deep learning model (LSTM) processes the news text and predicts sentiment scores.

    #### Use Cases

    - Financial analysts can quickly gauge market sentiment from news feeds
    - Investors can prioritize important financial news
    - Researchers can analyze sentiment trends in financial markets

    #### Technologies Used

    - **Backend**: FastAPI, TensorFlow, NLTK
    - **Frontend**: Streamlit
    - **Models**: LSTM with BERT embeddings
    """)
    
    # Show API documentation link
    st.markdown(f"**API Documentation**: [Swagger UI]({API_URL}/docs)")