"""
Unit tests for the Predictor class.

Tests inference including normalization, OOV mapping, and ranking.
Extra credit module.
"""


def test_predict_next_returns_k_results():
    """Test that predict_next() returns exactly k predictions."""
    pass


def test_predict_next_sorted_by_probability():
    """Test that results are sorted by probability (highest first)."""
    pass


def test_predict_next_handles_oov():
    """Test that predict_next() handles all-OOV context without crashing."""
    pass


def test_map_oov_replaces_unknown():
    """Test that map_oov() replaces unknown words with <UNK>."""
    pass


def test_map_oov_keeps_known():
    """Test that map_oov() leaves known words unchanged."""
    pass
