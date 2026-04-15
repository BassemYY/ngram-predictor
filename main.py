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


if __name__ == "__main__":
    main()
