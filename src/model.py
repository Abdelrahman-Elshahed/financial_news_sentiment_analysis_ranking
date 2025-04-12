import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, LayerNormalization,
    Input
)
from tensorflow.keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

def build_lstm_model(vocab_size, embedding_dim, max_length, embedding_matrix=None):

    model = Sequential([
        # Embedding layer - with BERT embeddings
        Embedding(input_dim=vocab_size,  # This now matches the embedding_matrix size
                output_dim=embedding_dim,
                weights=[embedding_matrix] if embedding_matrix is not None else None,
                input_length=max_length,
                trainable=False),  # Keep BERT embeddings fixed
        
        # Dimensionality reduction since BERT embeddings are large
        Dense(128),  # Project down to a smaller dimension
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LayerNormalization(),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        LayerNormalization(),
        
        # Dense layers
        Dense(32, activation='relu'),
        Dense(3, activation='linear')  # 3 outputs for pos, neg, neu
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_binary_lstm_model(vocab_size, embedding_dim, max_length, embedding_matrix=None):

    # Input layer
    inputs = Input(shape=(max_length,))
    
    # Embedding layer
    if embedding_matrix is not None:
        x = Embedding(input_dim=vocab_size, 
                     output_dim=embedding_dim,
                     weights=[embedding_matrix],
                     input_length=max_length,
                     trainable=False)(inputs)
    else:
        x = Embedding(input_dim=vocab_size, 
                     output_dim=embedding_dim,
                     input_length=max_length)(inputs)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(LSTM(32))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)
    
    # Output layer for binary sentiment (positive vs negative)
    outputs = Dense(2, activation='softmax', name='sentiment_output')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_bert_model(max_length=128):

    from transformers import TFBertModel, BertConfig
    
    # Load BERT model configuration
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    # BERT layer
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    # Input layers
    input_ids = Input(shape=(max_length,), dtype='int32', name='input_ids')
    attention_mask = Input(shape=(max_length,), dtype='int32', name='attention_mask')
    
    # Get BERT outputs
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)[0]
    
    # Take [CLS] token output
    cls_output = bert_outputs[:, 0, :]
    
    # Sentiment prediction head
    x = Dense(128, activation='relu')(cls_output)
    x = Dropout(0.2)(x)
    sentiment_output = Dense(3, activation='linear', name='sentiment')(x)  # pos, neg, neu
    
    # Ranking head (optional)
    x_rank = Dense(64, activation='relu')(cls_output)
    x_rank = Dropout(0.2)(x_rank)
    rank_output = Dense(1, activation='linear', name='rank')(x_rank)
    
    # Create and compile multi-output model
    model = Model(inputs=[input_ids, attention_mask], 
                  outputs=[sentiment_output, rank_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss={
            'sentiment': 'mse',
            'rank': 'mse'
        },
        metrics={
            'sentiment': 'mae',
            'rank': 'mae'
        }
    )
    
    return model

def build_logistic_regression_model():

    # Create a pipeline with TF-IDF vectorizer and Logistic Regression
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('lr', MultiOutputRegressor(LogisticRegression(max_iter=1000, C=1.0)))
    ])
    
    return model

def build_sentiment_model(vocab_size, embedding_dim, max_length, lstm_units=128, dropout_rate=0.2, embedding_matrix=None):

    import tensorflow as tf
    
    # Input layer
    inputs = tf.keras.Input(shape=(max_length,))
    
    # Embedding layer
    if embedding_matrix is not None:
        x = tf.keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False
        )(inputs)
    else:
        x = tf.keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim
        )(inputs)
    
    # Bidirectional LSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units // 2)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer for binary sentiment (positive vs negative)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model
