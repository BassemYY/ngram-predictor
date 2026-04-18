# N-Gram Next-Word Predictor

A statistical next-word prediction system built on N-Gram language models. The project processes raw Project Gutenberg text corpora, builds a smoothed N-Gram model with backoff, and exposes an interactive CLI where the user types a phrase and receives the top-K predicted next words. All pipeline steps are driven through a single entry point (`main.py`) with configurable parameters loaded from `config/.env`.

---

## Requirements

- Python 3.10 or higher
- All third-party dependencies are listed in `requirements.txt`

---

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd ngram-predictor
   ```

2. **Create and activate a conda environment**
   ```bash
   conda create -n ngram-env python=3.10
   conda activate ngram-env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Populate `config/.env`**  
   Create `config/.env` with the following variables (adjust paths if needed):
   ```
   TRAIN_RAW_DIR=data/raw/train/
   EVAL_RAW_DIR=data/raw/eval/
   TRAIN_TOKENS=data/processed/train_tokens.txt
   EVAL_TOKENS=data/processed/eval_tokens.txt
   MODEL=data/model/model.json
   VOCAB=data/model/vocab.json
   UNK_THRESHOLD=3
   TOP_K=3
   NGRAM_ORDER=4
   ```

5. **Download raw text files**  
   Download the following Project Gutenberg books and place them in the indicated folders:

   | File | Title | Folder |
   |------|-------|--------|
   | `108-0.txt` | Franklin's Autobiography | `data/raw/train/` |
   | `1661-0.txt` | The Adventures of Sherlock Holmes | `data/raw/train/` |
   | `2852-0.txt` | The Hound of the Baskervilles | `data/raw/train/` |
   | `834-0.txt` | The Return of Sherlock Holmes | `data/raw/train/` |
   | `3289-0.txt` | The Valley of Fear | `data/raw/eval/` |

   Download from: `https://www.gutenberg.org/files/<ID>/<ID>-0.txt`

---

## Usage

Run each pipeline step via `main.py --step`:

```bash
# Step 1 — Data preparation (tokenize and save corpus)
python main.py --step dataprep

# Step 2 — Train the N-Gram model
python main.py --step model

# Step 3 — Interactive next-word prediction CLI
python main.py --step inference

# Step 4 — Evaluate model perplexity on held-out corpus
python main.py --step evaluate

# Run full pipeline (dataprep → model → inference)
python main.py --step all
```

---

## Project Structure

```
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/          # Raw training .txt files
│   │   └── eval/           # Raw evaluation .txt files
│   ├── processed/          # Tokenized output files
│   └── model/              # Saved model and vocab JSON files
├── src/
│   ├── data_prep/
│   │   └── normalizer.py   # Normalizer class (M1)
│   ├── model/
│   │   └── ngram_model.py  # NGramModel class (M2)
│   ├── inference/
│   │   └── predictor.py    # Predictor class (M3)
│   ├── evaluation/
│   │   └── evaluator.py    # Evaluator class (extra credit)
│   └── ui/
│       └── app.py          # PredictorUI class (extra credit)
├── tests/
│   ├── test_normalizer.py
│   ├── test_ngram_model.py
│   └── test_predictor.py
├── main.py
├── requirements.txt
└── README.md
```

