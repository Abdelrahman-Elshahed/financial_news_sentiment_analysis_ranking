import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def rank_news_by_financial_importance(news_texts, sentiment_predictions):

    # Calculate financial importance (positive - negative + 0.5*neutral)
    financial_importance = sentiment_predictions[:, 0] - sentiment_predictions[:, 1] + 0.5 * sentiment_predictions[:, 2]
    
    # Create DataFrame with news and importance scores
    ranked_df = pd.DataFrame({
        'news_text': news_texts.values if hasattr(news_texts, 'values') else news_texts,
        'positive_score': sentiment_predictions[:, 0],
        'negative_score': sentiment_predictions[:, 1],
        'neutral_score': sentiment_predictions[:, 2],
        'financial_importance': financial_importance
    })
    
    # Remove duplicates before ranking
    ranked_df = ranked_df.drop_duplicates(subset=['news_text'])
    
    # Sort by importance score in descending order
    ranked_df = ranked_df.sort_values(by='financial_importance', ascending=False)
    
    return ranked_df

def rank_news_by_binary_sentiment(news_texts, binary_sentiment_predictions):

    # Calculate sentiment score (positive - negative)
    sentiment_score = binary_sentiment_predictions[:, 0] - binary_sentiment_predictions[:, 1]
    
    # Create DataFrame with news and scores
    ranked_df = pd.DataFrame({
        'news_text': news_texts.values if hasattr(news_texts, 'values') else news_texts,
        'positive_score': binary_sentiment_predictions[:, 0],
        'negative_score': binary_sentiment_predictions[:, 1],
        'sentiment_score': sentiment_score
    })
    
    # Remove duplicates before ranking
    ranked_df = ranked_df.drop_duplicates(subset=['news_text'])
    
    # Sort by sentiment score in descending order (most positive first)
    ranked_df = ranked_df.sort_values(by='sentiment_score', ascending=False)
    
    return ranked_df

def visualize_rankings(ranked_news, n=10):

    if n <= 0 or n > len(ranked_news):
        n = min(10, len(ranked_news))
    
    # Get top and bottom n news
    top_n = ranked_news.head(n)
    bottom_n = ranked_news.tail(n)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot top n
    x_top = np.arange(len(top_n))
    importance_col = 'financial_importance' if 'financial_importance' in ranked_news.columns else 'sentiment_score'
    
    ax1.barh(x_top, top_n[importance_col], color='green')
    ax1.set_yticks(x_top)
    ax1.set_yticklabels([t[:50] + '...' for t in top_n['news_text']])
    ax1.set_title(f'Top {n} Most Important News')
    ax1.set_xlabel(importance_col.replace('_', ' ').title())
    
    # Plot bottom n
    x_bottom = np.arange(len(bottom_n))
    ax2.barh(x_bottom, bottom_n[importance_col], color='red')
    ax2.set_yticks(x_bottom)
    ax2.set_yticklabels([t[:50] + '...' for t in bottom_n['news_text']])
    ax2.set_title(f'Bottom {n} Least Important News')
    ax2.set_xlabel(importance_col.replace('_', ' ').title())
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/ranked_news.png'
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def get_top_news(ranked_news, n=10, include_scores=False):

    if n <= 0 or n > len(ranked_news):
        n = len(ranked_news)
    
    # Get top n rows
    top_n = ranked_news.head(n)
    
    if include_scores:
        return top_n
    else:
        return top_n['news_text'].tolist()

def get_bottom_news(ranked_news, n=10, include_scores=False):

    if n <= 0 or n > len(ranked_news):
        n = len(ranked_news)
    
    # Get bottom n rows
    bottom_n = ranked_news.tail(n)
    
    if include_scores:
        return bottom_n
    else:
        return bottom_n['news_text'].tolist()

def export_ranked_news(ranked_news, filepath='ranked_news.csv'):

    try:
        ranked_news.to_csv(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error exporting ranked news: {e}")
        return False
