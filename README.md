# Sentiment CSV Annotator

Append Hugging Face sentiment predictions to CSV files that contain a `sentence` column (or another text column you specify).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python sentiment_csv.py data/input.csv
```

Flags:

- `--model`: model repo or path (default `cardiffnlp/twitter-roberta-base-sentiment-latest`)
- `--text-column`: name of the column with text (default `sentence`)
- `--batch-size`: adjust for speed vs. memory (default 16)
- `--output-dir`: where to write outputs (defaults to input file directory)
- `--suffix`: inserted before the file extension when not overwriting (default `_sentiment`)
- `--overwrite`: replace the input file instead of writing a new one
- `--device`: `auto` (default), `cpu`, or `cuda`

Outputs mirror the input CSV plus two columns:

- `sentiment_label`
- `sentiment_score` (confidence for the predicted label)

## Notes

- The script will fail fast if the chosen text column is missing.
- Rows with empty/missing text keep `sentiment_*` columns empty.
- For different models, pass `--model` and the classifier will download as needed.
