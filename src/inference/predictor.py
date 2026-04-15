"""
Predictor module for next-word inference.

This module takes a pre-loaded NGramModel and Normalizer, normalizes input text,
extracts context, maps out-of-vocabulary words, and returns the top-k most likely
next words sorted by probability. Backoff lookup is delegated to NGramModel.
"""

from typing import List


class Predictor:
    """
    Predicts the top-k next words given input text.
    
    Accepts a pre-loaded NGramModel and Normalizer via the constructor,
    normalizes input text, extracts context, maps OOV words, and returns
    the top-k predicted next words sorted by probability. Backoff lookup
    is delegated to NGramModel.lookup().
    """

    def __init__(self, model, normalizer):
        """
        Initialize Predictor with pre-loaded model and normalizer.
        
        Args:
            model: Pre-loaded NGramModel instance (not loaded from file).
            normalizer: Pre-loaded Normalizer instance.
        """
        pass

    def normalize(self, text: str) -> List[str]:
        """
        Normalize input text and extract context.
        
        Calls Normalizer.normalize(text) and extracts the last NGRAM_ORDER-1
        words as context for lookup.
        
        Args:
            text: Input text string to normalize.
            
        Returns:
            List of context words (length ≤ NGRAM_ORDER-1).
        """
        pass

    def map_oov(self, context: List[str]) -> List[str]:
        """
        Replace out-of-vocabulary words with <UNK>.
        
        Args:
            context: List of context words.
            
        Returns:
            Context list with OOV words replaced by <UNK>.
        """
        pass

    def predict_next(self, text: str, k: int) -> List[str]:
        """
        Predict top-k next words given input text.
        
        Orchestrates: normalize → extract context → map OOV → lookup →
        sort by probability → return top-k.
        
        Args:
            text: Input text string from user.
            k: Number of top predictions to return.
            
        Returns:
            List of up to k predicted words sorted by probability (highest first).
            Returns empty list if no predictions found at any order.
        """
        pass


def main():
    """
    Entry point for Predictor module.
    
    Demonstrates predicting next words for sample input.
    """
    print("Predictor module initialized.")


if __name__ == "__main__":
    main()
