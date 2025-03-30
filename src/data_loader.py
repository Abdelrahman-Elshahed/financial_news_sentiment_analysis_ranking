import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def convert_sentiment_to_binary(df):

    df['sentiment_label'] = df['sentiment'].map({'NEGATIVE': 0, 'POSITIVE': 1})
    return df

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):

    from sklearn.model_selection import train_test_split
    
    # First split: separate out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Calculate the adjusted validation size for the remaining data
    adjusted_val_size = val_size / (1 - test_size)
    
    # Second split: separate the training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def extract_sentiment_components(y_train, y_val, y_test):

    y_train_pos, y_train_neg, y_train_neu = y_train["pos"], y_train["neg"], y_train["neu"]
    y_val_pos, y_val_neg, y_val_neu = y_val["pos"], y_val["neg"], y_val["neu"]
    y_test_pos, y_test_neg, y_test_neu = y_test["pos"], y_test["neg"], y_test["neu"]
    
    return (y_train_pos, y_train_neg, y_train_neu, 
            y_val_pos, y_val_neg, y_val_neu,
            y_test_pos, y_test_neg, y_test_neu)

def apply_min_max_scaling(train_data, val_data=None, test_data=None, fit_on_train=True):

    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    # Initialize the scaler
    if fit_on_train:
        scaler = MinMaxScaler()
        
        # Reshape if it's a 1D array
        if len(train_data.shape) == 1:
            train_data = train_data.reshape(-1, 1)
            
        train_scaled = scaler.fit_transform(train_data)
    else:
        scaler = None
        train_scaled = train_data
        
    # Scale validation data if provided
    if val_data is not None:
        if len(val_data.shape) == 1:
            val_data = val_data.reshape(-1, 1)
        val_scaled = scaler.transform(val_data)
    else:
        val_scaled = None
        
    # Scale test data if provided
    if test_data is not None:
        if len(test_data.shape) == 1:
            test_data = test_data.reshape(-1, 1)
        test_scaled = scaler.transform(test_data)
    else:
        test_scaled = None
        
    return train_scaled, val_scaled, test_scaled, scaler
