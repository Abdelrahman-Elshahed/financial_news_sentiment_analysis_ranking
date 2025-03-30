import unittest
import sys
import os
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_text, tokenize_and_remove_stopwords, tokenize_and_pad_sequences

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_text(self):
        # Test basic text preprocessing
        text = "Hello, World! This is a TEST with 123 numbers."
        processed = preprocess_text(text)
        
        # Should be lowercase with no punctuation
        self.assertEqual(processed, "hello world this is a test with 123 numbers")
        
        # Test handling of accented characters
        text_with_accents = "Café éléphant résumé"
        processed = preprocess_text(text_with_accents)
        self.assertEqual(processed, "cafe elephant resume")
        
    def test_tokenize_and_remove_stopwords(self):
        # Test stopword removal
        text = "this is a sample text with some common stopwords"
        processed = tokenize_and_remove_stopwords(text)
        
        # Common stopwords should be removed
        self.assertNotIn("this", processed)
        self.assertNotIn("is", processed)
        self.assertNotIn("with", processed)
        self.assertNotIn("some", processed)
        
        # Content words should remain
        self.assertIn("sample", processed)
        self.assertIn("text", processed)
        self.assertIn("common", processed)
        self.assertIn("stopwords", processed)
        
    def test_tokenize_and_pad_sequences(self):
        # Test sequence tokenization and padding
        texts = ["this is first document", "this is second"]
        max_length = 5
        
        # First call should create and fit tokenizer
        padded_sequences, tokenizer = tokenize_and_pad_sequences(texts, max_length)
        
        # Check shapes
        self.assertEqual(padded_sequences.shape, (2, max_length))
        
        # Test padding (short sequence should be padded with zeros)
        self.assertEqual(padded_sequences[1, 3], 0)  # Padding in last position for shorter text
        
        # Test reusing tokenizer
        new_texts = ["this is third document"]
        padded_seq2, _ = tokenize_and_pad_sequences(new_texts, max_length, tokenizer, fit_on_texts=False)
        
        # Words seen before should have the same token IDs
        self.assertEqual(padded_seq2[0, 0], padded_sequences[0, 0])  # "this" should have same ID
        
        # Test truncation
        long_text = ["this is a very long text that should be truncated"]
        padded_seq3, _ = tokenize_and_pad_sequences(long_text, max_length, tokenizer, fit_on_texts=False)
        self.assertEqual(padded_seq3.shape, (1, max_length))

if __name__ == '__main__':
    unittest.main()