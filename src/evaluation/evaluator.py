"""
Model Evaluator module for computing perplexity on a held-out corpus.

This module computes cross-entropy and perplexity on a held-out evaluation corpus
using the trained n-gram model. It is optional and counts as extra credit (+5 points).
"""

import math
from typing import Optional


class Evaluator:
    """
    Evaluates n-gram model perplexity on a held-out corpus.
    
    Computes cross-entropy and perplexity by scoring each word in the
    evaluation corpus given its context via backoff lookup.
    This is an optional extra credit module.
    """

    def __init__(self, model, normalizer):
        """
        Initialize Evaluator with pre-loaded model and normalizer.
        
        Args:
            model: Pre-loaded NGramModel instance.
            normalizer: Pre-loaded Normalizer instance.
        """
        pass

    def score_word(self, word: str, context: list) -> Optional[float]:
        """
        Compute log₂ P(word | context) using backoff lookup.
        
        Args:
            word: Target word to score.
            context: Context words for lookup.
            
        Returns:
            log₂ P(word | context), or None if zero probability at all orders.
        """
        pass

    def compute_perplexity(self, eval_file: str) -> tuple:
        """
        Compute perplexity over the full evaluation corpus.
        
        Iterates through each word in the corpus, scores it using backoff,
        accumulates log probabilities, and computes final perplexity.
        
        Args:
            eval_file: Path to tokenized evaluation file.
            
        Returns:
            Tuple of (perplexity: float, num_evaluated: int, num_skipped: int).
        """
        pass

    def run(self, eval_file: str) -> None:
        """
        Orchestrate perplexity computation and print results.
        
        Args:
            eval_file: Path to tokenized evaluation file.
        """
        pass


def main():
    """
    Entry point for Evaluator module.
    
    Demonstrates computing perplexity on sample corpus.
    """
    print("Evaluator module initialized.")


if __name__ == "__main__":
    main()
