"""
Unit tests for the NGramModel class.

Tests model building, probability computation, and backoff lookup.
Extra credit module.
"""


def test_build_vocab_replaces_low_frequency():
    """Test that build_vocab() replaces low-frequency words with <UNK>."""
    pass


def test_lookup_seen_context():
    """Test that lookup() returns predictions for a seen context."""
    pass


def test_lookup_unseen_context():
    """Test that lookup() falls back to unigram for unseen context."""
    pass


def test_lookup_probabilities_sum():
    """Test that probabilities for any context sum to approximately 1."""
    pass


def test_save_and_load_model():
    """Test that save/load preserves model structure."""
    pass
