"""
Predictor module for next-word inference.

Accepts a pre-loaded NGramModel and Normalizer via the constructor, normalizes
input text, maps OOV words to <UNK>, delegates backoff lookup to NGramModel,
and returns the top-k predicted next words sorted by probability.
"""

import os
from typing import List


class Predictor:
    """
    Predicts the top-k next words given input text.

    Receives a pre-loaded NGramModel and Normalizer via dependency injection.
    Normalizes input, maps OOV context words to <UNK>, and delegates all
    backoff lookup logic to NGramModel.lookup().
    """

    def __init__(self, model, normalizer):
        """
        Initialize Predictor with a pre-loaded model and normalizer.

        Args:
            model: Pre-loaded NGramModel instance.
            normalizer: Pre-loaded Normalizer instance.
        """
        self.model = model
        self.normalizer = normalizer
        self.order = int(os.environ.get("NGRAM_ORDER", "4"))

    def normalize(self, text: str) -> List[str]:
        """
        Normalize input text and extract the last NGRAM_ORDER-1 words as context.

        Calls Normalizer.normalize() for consistent processing, then returns
        only the last (NGRAM_ORDER - 1) tokens for use as lookup context.

        Args:
            text: Raw input text string from the user.

        Returns:
            List of up to NGRAM_ORDER-1 context words.
        """
        cleaned = self.normalizer.normalize(text)
        tokens = [t for t in cleaned.split() if t]
        keep = self.order - 1
        return tokens[-keep:] if keep > 0 else []

    def map_oov(self, context: List[str]) -> List[str]:
        """
        Replace out-of-vocabulary words in context with <UNK>.

        Args:
            context: List of context words.

        Returns:
            Context with any unknown words replaced by <UNK>.
        """
        return [w if w in self.model.vocab_set else "<UNK>" for w in context]

    def predict_next(self, text: str, k: int) -> List[str]:
        """
        Predict the top-k next words given input text.

        Orchestrates: normalize → map_oov → NGramModel.lookup() →
        sort by probability (highest first) → return top-k words.

        Args:
            text: Raw input text string from the user.
            k: Number of top predictions to return.

        Returns:
            List of up to k predicted words sorted by probability (highest first).
            Returns empty list if input is empty or no predictions found.
        """
        context = self.normalize(text)
        if not context:
            return []

        mapped = self.map_oov(context)
        distribution = self.model.lookup(mapped)
        if not distribution:
            return []

        ranked = sorted(distribution.items(), key=lambda x: (-x[1], x[0]))
        return [word for word, _ in ranked[:k]]


def main():
    """
    Entry point for Predictor module.

    Loads model and runs the interactive next-word prediction CLI loop standalone.
    """
    import os
    from dotenv import load_dotenv
    from src.model.ngram_model import NGramModel
    from src.data_prep.normalizer import Normalizer

    load_dotenv(dotenv_path="config/.env")

    model_path = os.environ.get("MODEL", "data/model/model.json")
    vocab_path = os.environ.get("VOCAB", "data/model/vocab.json")
    top_k      = int(os.environ.get("TOP_K", "3"))

    normalizer = Normalizer()
    model = NGramModel()
    model.load(model_path, vocab_path)
    predictor = Predictor(model, normalizer)

    print("Model loaded. Vocab size: " + str(len(model.vocab)))
    print("Type a phrase for top-" + str(top_k) + " predictions. Type 'quit' to exit.")
    print()

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        predictions = predictor.predict_next(text, top_k)
        print("Predictions: " + (", ".join(predictions) if predictions else "none"))
        print()


if __name__ == "__main__":
    main()

