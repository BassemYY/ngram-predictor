"""
Main entry point for the N-Gram Next-Word Predictor.

This module orchestrates the entire pipeline:
- Loads configuration from config/.env
- Instantiates Normalizer, NGramModel, and Predictor via dependency injection
- Provides a CLI with --step argument to run individual pipeline steps or the full pipeline
- Implements interactive inference loop for next-word prediction

Usage:
    python main.py --step dataprep    # Run data preparation
    python main.py --step model       # Run model training
    python main.py --step inference   # Run interactive CLI
    python main.py --step all         # Run full pipeline
    python main.py --step evaluate    # Run model evaluator (extra credit)
"""

import argparse
import os
import sys
from dotenv import load_dotenv


def run_dataprep():
    """Run the data preparation pipeline for training corpus."""
    from src.data_prep.normalizer import Normalizer

    train_raw_dir  = os.environ.get("TRAIN_RAW_DIR",  "data/raw/train/")
    eval_raw_dir   = os.environ.get("EVAL_RAW_DIR",   "data/raw/eval/")
    train_tokens   = os.environ.get("TRAIN_TOKENS",   "data/processed/train_tokens.txt")
    eval_tokens    = os.environ.get("EVAL_TOKENS",    "data/processed/eval_tokens.txt")

    n = Normalizer()

    for label, raw_dir, out_path in [
        ("train", train_raw_dir, train_tokens),
        ("eval",  eval_raw_dir,  eval_tokens),
    ]:
        print(f"[{label}] Loading raw text from {raw_dir} ...")
        raw = n.load(raw_dir)
        print(f"[{label}] Raw chars: {len(raw):,}")

        stripped = n.strip_gutenberg(raw)
        print(f"[{label}] Stripped chars: {len(stripped):,}")

        sentences = n.sentence_tokenize(stripped)
        print(f"[{label}] Sentences: {len(sentences):,}")

        tokenized = []
        for sent in sentences:
            normed = n.normalize(sent)
            tokens = n.word_tokenize(normed)
            if tokens:
                tokenized.append(tokens)

        total_tokens = sum(len(t) for t in tokenized)
        print(f"[{label}] Tokenized sentences: {len(tokenized):,}  |  Total tokens: {total_tokens:,}")

        n.save(tokenized, out_path)
        print(f"[{label}] Saved → {out_path}")
        print()


def run_model():
    """Run the model training pipeline: build vocab, counts, probabilities, save."""
    from src.model.ngram_model import NGramModel

    train_tokens = os.environ.get("TRAIN_TOKENS", "data/processed/train_tokens.txt")
    model_path   = os.environ.get("MODEL",        "data/model/model.json")
    vocab_path   = os.environ.get("VOCAB",        "data/model/vocab.json")

    m = NGramModel()

    print(f"[model] Building vocabulary from {train_tokens} ...")
    m.build_vocab(train_tokens)
    print(f"[model] Vocab size: {len(m.vocab):,} words")

    print(f"[model] Building n-gram counts and probabilities (order={m.order}) ...")
    m.build_counts_and_probabilities(train_tokens)
    total_contexts = sum(len(m.model[k]) for k in m.model)
    print(f"[model] Total contexts across all orders: {total_contexts:,}")

    m.save_model(model_path)
    print(f"[model] Saved model → {model_path}")
    m.save_vocab(vocab_path)
    print(f"[model] Saved vocab → {vocab_path}")
    print()


def main():
    """
    Main entry point.

    Loads environment configuration, parses CLI arguments, and orchestrates
    the pipeline execution based on the --step argument.
    """
    # Load environment variables first (before any other operations)
    load_dotenv(dotenv_path="config/.env")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="N-Gram Next-Word Predictor"
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "evaluate", "all"],
        default="all",
        help="Pipeline step to run (default: all)"
    )
    args = parser.parse_args()
    
    print(f"N-Gram Next-Word Predictor")
    print(f"Step: {args.step}")
    print()

    if args.step in ("dataprep", "all"):
        run_dataprep()

    if args.step in ("model", "all"):
        run_model()


if __name__ == "__main__":
    main()
