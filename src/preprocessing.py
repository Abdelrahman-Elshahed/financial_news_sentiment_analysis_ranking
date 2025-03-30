import re
import nltk
import unidecode
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define a function to get stopwords to ensure consistency
def get_stopwords():
    # Make sure stopwords are downloaded
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    return stop_words

# Get stopwords once at module level
STOP_WORDS = get_stopwords()

def preprocess_text(text):
    """Lowercase the text and remove special characters, numbers, and punctuation"""
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Remove accents/diacritics (for language normalization)
    text = unidecode.unidecode(text)
    
    return text
    
def tokenize_and_remove_stopwords(text):
    """Tokenize text (split by spaces) and remove stopwords"""
    # Make sure we have a lowercase string for consistent comparisons
    text = text.lower()
    
    # Split into tokens
    tokens = text.split()
    
    # Filter out stopwords using the module-level STOP_WORDS
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    
    # Join back into a string
    return ' '.join(filtered_tokens)

def tokenize_and_pad_sequences(texts, max_length, tokenizer=None, fit_on_texts=True):
    """Tokenize texts into sequences and pad them to a fixed length"""
    if fit_on_texts:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    elif tokenizer is None:
        raise ValueError("When fit_on_texts is False, you must provide a tokenizer")
        
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def apply_preprocessing_pipeline(X_train, X_val, X_test):
    """Apply full preprocessing pipeline to train, validation, and test sets"""
    # Apply basic preprocessing
    X_train = X_train.apply(preprocess_text)
    X_val = X_val.apply(preprocess_text)
    X_test = X_test.apply(preprocess_text)
    
    # Tokenize and remove stopwords
    X_train = X_train.apply(tokenize_and_remove_stopwords)
    X_val = X_val.apply(tokenize_and_remove_stopwords)
    X_test = X_test.apply(tokenize_and_remove_stopwords)
    
    return X_train, X_val, X_test


def create_bert_embedding_matrix_optimized(word_index, max_words=25000, embedding_dim=768, batch_size=64):

    import numpy as np
    import torch
    import os
    from transformers import BertTokenizer, BertModel
    
    print(f"Loading BERT model: bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Take only the most frequent words (those with lowest indices)
    # This dramatically reduces computation time while covering most common words
    vocab_size = min(len(word_index) + 1, max_words + 1)
    filtered_word_index = {word: idx for word, idx in word_index.items() if idx < max_words}
    
    print(f"Processing {len(filtered_word_index)} most frequent words out of {len(word_index)} total words")
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Process in batches for efficiency
    words = list(filtered_word_index.keys())
    indices = list(filtered_word_index.values())
    
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i+batch_size]
        batch_indices = indices[i:i+batch_size]
        
        # Encode all words in batch
        encoded_inputs = tokenizer(batch_words, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded_inputs)
            # Get [CLS] token embedding as word representation
            batch_embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy()
        
        # Store embeddings
        for j, idx in enumerate(batch_indices):
            embedding_matrix[idx] = batch_embeddings[j]
        
        # Save progress every 1000 words
        if i % 1000 == 0:
            print(f"Processed {i}/{len(filtered_word_index)} words")
            # Optionally save intermediate results
            os.makedirs('saved_models', exist_ok=True)
            np.save(f'saved_models/bert_embedding_matrix_partial_{i}.npy', embedding_matrix)
    
    print(f"Created BERT embeddings for {len(filtered_word_index)} words")
    return embedding_matrix
