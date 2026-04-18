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
        self.model = model
        self.normalizer = normalizer

    def score_word(self, word: str, context: list) -> Optional[float]:
        """
        Compute log₂ P(word | context) using backoff lookup.
        
        Args:
            word: Target word to score.
            context: Context words for lookup.
            
        Returns:
            log₂ P(word | context), or None if zero probability at all orders.
        """
        # Map OOV words to <unk>
        word_to_score = word if word in self.model.vocab else "<unk>"
        
        # Lookup probability using backoff
        prob = self.model.lookup(word_to_score, context)
        
        # Return None if zero probability, otherwise return log₂ probability
        if prob is None or prob == 0:
            return None
        
        return math.log2(prob)

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
        total_log_prob = 0.0
        num_evaluated = 0
        num_skipped = 0
        
        # Load tokenized sentences from eval file
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokens = line.split()
                    
                    # Score each word given its context
                    for i, word in enumerate(tokens):
                        # Build context: use up to (order-1) preceding words
                        context_start = max(0, i - (self.model.order - 1))
                        context = tokens[context_start:i]
                        
                        # Score this word
                        log_prob = self.score_word(word, context)
                        
                        if log_prob is not None:
                            total_log_prob += log_prob
                            num_evaluated += 1
                        else:
                            num_skipped += 1
        except FileNotFoundError:
            print(f"Error: {eval_file} not found.")
            return 0.0, 0, 0
        
        # Compute perplexity: 2^(-sum_log_prob / N)
        if num_evaluated > 0:
            perplexity = math.pow(2.0, -total_log_prob / num_evaluated)
        else:
            perplexity = float('inf')
        
        return perplexity, num_evaluated, num_skipped

    def run(self, eval_file: str) -> None:
        """
        Orchestrate perplexity computation and print results.
        
        Args:
            eval_file: Path to tokenized evaluation file.
        """
        perplexity, num_evaluated, num_skipped = self.compute_perplexity(eval_file)
        
        print(f"[evaluate] Perplexity: {perplexity:.2f}")
        print(f"[evaluate] Evaluated: {num_evaluated:,} words, Skipped: {num_skipped:,} words")


def main():
    """
    Entry point for Evaluator module.
    
    Demonstrates computing perplexity on sample corpus.
    """
    print("Evaluator module initialized.")


if __name__ == "__main__":
    main()
