"""
Normalizer module for data preparation.

This module is responsible for loading, cleaning, tokenizing, and saving text corpora
from Project Gutenberg. The Normalizer class handles text normalization (lowercasing,
punctuation removal, number removal, whitespace cleaning) and tokenization at both
sentence and word levels. It is used in two contexts: in Module 1 (Data Prep) to
process whole raw files, and in Module 3 (Inference) to normalize single input strings.
"""

import os
from typing import List


class Normalizer:
    """
    Handles text loading, cleaning, tokenization, and saving.
    
    Provides methods to load raw text files, strip Gutenberg headers/footers,
    normalize text (lowercase, remove punctuation/numbers/whitespace), and
    tokenize into sentences and words. The same normalize() method is called
    by both data prep and inference modules to ensure consistent processing.
    """

    def load(self, folder_path: str) -> str:
        """
        Load all .txt files from a folder and concatenate them.
        
        Args:
            folder_path: Path to folder containing .txt files.
            
        Returns:
            Concatenated text from all .txt files in the folder.
            
        Raises:
            FileNotFoundError: If folder does not exist.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Folder not found: {folder_path}. Check TRAIN_RAW_DIR in config/.env."
            )
        txt_files = sorted(os.listdir(folder_path))
        all_text = []
        for filename in txt_files:
            if filename.endswith(".txt"):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    all_text.append(f.read())
        return "\n".join(all_text)

    def strip_gutenberg(self, text: str) -> str:
        """
        Remove Project Gutenberg header and footer markers.
        
        Removes all text before and including the START marker, and all text
        from and including the END marker.
        
        Args:
            text: Raw text containing Gutenberg markers.
            
        Returns:
            Text with headers and footers removed.
        """
        import re
        # Extract only the content between each START and END marker pair
        chunks = re.findall(
            r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\*]*\*\*\*(.*?)\*\*\* END OF THE PROJECT GUTENBERG EBOOK",
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        return "\n".join(chunk.strip() for chunk in chunks)

    def lowercase(self, text: str) -> str:
        """
        Convert all text to lowercase.
        
        Args:
            text: Input text.
            
        Returns:
            Lowercased text.
        """
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """
        Remove all punctuation characters.
        
        Args:
            text: Input text.
            
        Returns:
            Text with punctuation removed.
        """
        import re
        return re.sub(r"[^\w\s]|_", "", text)

    def remove_numbers(self, text: str) -> str:
        """
        Remove all numeric digits.
        
        Args:
            text: Input text.
            
        Returns:
            Text with numbers removed.
        """
        import re
        return re.sub(r"\d+", "", text)

    def remove_whitespace(self, text: str) -> str:
        """
        Remove extra whitespace and blank lines.
        
        Collapses multiple spaces into single spaces and removes blank lines.
        
        Args:
            text: Input text.
            
        Returns:
            Text with normalized whitespace.
        """
        import re
        text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
        text = re.sub(r"\n\s*\n+", "\n", text)  # collapse blank lines
        return text.strip()

    def normalize(self, text: str) -> str:
        """
        Apply all normalization steps in order.
        
        Applies: lowercase -> remove punctuation -> remove numbers -> remove whitespace.
        This is the single method that other modules call to normalize text consistently.
        
        Args:
            text: Input text to normalize.
            
        Returns:
            Normalized text.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text.
            
        Returns:
            List of sentences.
        """
        import nltk
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def word_tokenize(self, sentence: str) -> List[str]:
        """
        Split a sentence into tokens (words).
        
        Args:
            sentence: Input sentence.
            
        Returns:
            List of tokens separated by spaces; no empty tokens.
        """
        return [token for token in sentence.split() if token]

    def save(self, sentences: List[List[str]], filepath: str) -> None:
        """
        Write tokenized sentences to output file.
        
        Format: one sentence per line, tokens separated by spaces.
        
        Args:
            sentences: List of tokenized sentences (each sentence is a list of tokens).
            filepath: Path to output file.
        """
        pass


def main():
    """
    Entry point for Normalizer module.
    
    Demonstrates loading, normalizing, tokenizing, and saving a corpus.
    """
    print("Normalizer module initialized.")


if __name__ == "__main__":
    main()
