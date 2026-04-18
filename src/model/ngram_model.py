"""
NGramModel module placeholder.
"""


class NGramModel:
    """N-Gram language model (not yet implemented)."""

    def __init__(self):
        pass

    def build_vocab(self, token_file):
        pass

    def build_counts_and_probabilities(self, token_file):
        pass

    def lookup(self, context):
        pass

    def save_model(self, model_path):
        pass

    def save_vocab(self, vocab_path):
        pass

    def load(self, model_path, vocab_path):
        pass


def main():
    print("NGramModel module initialized.")


if __name__ == "__main__":
    main()
