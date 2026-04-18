"""
NGramModel module for building and storing n-gram probability tables.

Builds vocabulary with UNK thresholding, counts n-grams at all orders from 1
up to NGRAM_ORDER, computes MLE probabilities, and provides backoff lookup.
Supports saving and loading model/vocab as JSON files.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


class NGramModel:
    """
    Builds and stores n-gram probability tables with backoff lookup.

    Reads NGRAM_ORDER and UNK_THRESHOLD from environment variables.
    Manages vocabulary construction, n-gram counting at all orders,
    MLE probability computation, and backoff lookup. Supports JSON
    serialization for persistence between pipeline runs.
    """

    def __init__(self):
        """
        Initialize NGramModel.

        Reads NGRAM_ORDER and UNK_THRESHOLD from environment variables.
        Initializes empty vocab and model containers.
        """
        self.order = int(os.environ.get("NGRAM_ORDER", "4"))
        self.unk_threshold = int(os.environ.get("UNK_THRESHOLD", "3"))
        self.vocab: List[str] = []
        self.vocab_set: set = set()
        self.model: Dict[str, Dict[str, Dict[str, float]]] = {}

    def build_vocab(self, token_file: str) -> None:
        """
        Build vocabulary from a tokenized file.

        Reads all tokens, keeps words appearing >= UNK_THRESHOLD times,
        and adds <UNK> to represent rare/unseen words.

        Args:
            token_file: Path to tokenized file (one sentence per line,
                        tokens space-separated).
        """
        counts: Counter = Counter()
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                counts.update(tokens)

        vocab = {word for word, count in counts.items() if count >= self.unk_threshold}
        vocab.add("<UNK>")

        self.vocab = sorted(vocab)
        self.vocab_set = set(self.vocab)

    def build_counts_and_probabilities(self, token_file: str) -> None:
        """
        Build n-gram counts at all orders and compute MLE probabilities.

        Reads the tokenized file, maps rare words to <UNK>, slides a window
        across every sentence to count all n-grams from 1-gram up to NGRAM_ORDER.
        Computes MLE probabilities immediately after counting to avoid hidden
        ordering bugs.

        Storage format:
            model["1gram"]  = {word: probability}
            model["ngram"]  = {" ".join(context): {word: probability}}  for n >= 2

        Args:
            token_file: Path to tokenized file (one sentence per line).
        """
        if not self.vocab_set:
            self.build_vocab(token_file)

        counts_1: Counter = Counter()
        counts_n: Dict[int, Dict[str, Counter]] = {
            n: defaultdict(Counter) for n in range(2, self.order + 1)
        }

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                raw_tokens = line.strip().split()
                if not raw_tokens:
                    continue
                tokens = [t if t in self.vocab_set else "<UNK>" for t in raw_tokens]

                counts_1.update(tokens)

                for n in range(2, self.order + 1):
                    for i in range(n - 1, len(tokens)):
                        context_key = " ".join(tokens[i - n + 1 : i])
                        counts_n[n][context_key][tokens[i]] += 1

        model: Dict[str, object] = {}

        total = sum(counts_1.values())
        model["1gram"] = {word: count / total for word, count in counts_1.items()}

        for n in range(2, self.order + 1):
            gram_key = f"{n}gram"
            model[gram_key] = {}
            for context_key, word_counts in counts_n[n].items():
                total_n = sum(word_counts.values())
                model[gram_key][context_key] = {
                    word: count / total_n for word, count in word_counts.items()
                }

        self.model = model

    def lookup(self, context: List[str]) -> Dict[str, float]:
        """
        Backoff lookup: try highest-order context first, fall back to lower orders.

        Maps any OOV context words to <UNK>, then attempts a match starting at
        the highest possible n-gram order and stepping down to 1-gram. Returns
        the probability distribution from the first successful match.

        This is the single source of backoff logic in the project.

        Args:
            context: List of context words (last NGRAM_ORDER-1 words of input).

        Returns:
            Dict of {word: probability} from the highest-order successful match.
            Returns empty dict if no match is found at any order.
        """
        if not self.model:
            return {}

        mapped = [w if w in self.vocab_set else "<UNK>" for w in context]

        for n in range(self.order, 0, -1):
            gram_key = str(n) + "gram"
            if gram_key not in self.model:
                continue

            if n == 1:
                return dict(self.model["1gram"])

            needed = n - 1
            context_key = " ".join(mapped[-needed:])
            dist = self.model[gram_key].get(context_key)
            if dist:
                return dist

        return {}

    def save_model(self, model_path: str) -> None:
        """
        Save all probability tables to a JSON file.

        Creates parent directories if they don't exist. Writes one key per
        n-gram order (e.g. "1gram" through "4gram" for NGRAM_ORDER=4).

        Args:
            model_path: Path to output model.json file.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.model, f, ensure_ascii=True)

    def save_vocab(self, vocab_path: str) -> None:
        """
        Save the vocabulary list to a JSON file.

        Args:
            vocab_path: Path to output vocab.json file.
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=True)

    def load(self, model_path: str, vocab_path: str) -> None:
        """
        Load model and vocabulary from JSON files.

        Populates self.model, self.vocab, and self.vocab_set.
        Called once in main() before passing the model to Predictor.

        Args:
            model_path: Path to model.json file.
            vocab_path: Path to vocab.json file.

        Raises:
            FileNotFoundError: If model.json or vocab.json does not exist.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "model.json not found at " + model_path + ". Run the Model module first."
            )
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                "vocab.json not found at " + vocab_path + ". Run the Model module first."
            )
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.vocab_set = set(self.vocab)


def main():
    """Entry point for NGramModel module."""
    print("NGramModel module initialized.")


if __name__ == "__main__":
    main()

