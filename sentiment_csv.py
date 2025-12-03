#!/usr/bin/env python3
"""CLI to append Hugging Face sentiment predictions to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append sentiment predictions to CSV files. "
            "Assumes a text column named 'sentence' unless overridden."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to CSV file(s) to process.",
    )
    parser.add_argument(
        "--text-column",
        default="sentence",
        help="Name of the column that contains the text to score.",
    )
    parser.add_argument(
        "--model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="Hugging Face model repo or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference. Increase for speed if memory allows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write outputs to. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--suffix",
        default="_sentiment",
        help="Suffix to insert before the file extension for the output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file instead of writing a new one.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on. 'auto' picks GPU if available, otherwise CPU.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_dir: Path | None, suffix: str, overwrite: bool) -> Path:
    if overwrite:
        return input_path

    target_dir = output_dir if output_dir is not None else input_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    new_name = f"{stem}{suffix}{input_path.suffix}"
    return target_dir / new_name


def build_classifier(model_name: str, device_choice: str):
    if device_choice == "auto":
        device_index = 0 if torch.cuda.is_available() else -1
    elif device_choice == "cpu":
        device_index = -1
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device_index = 0

    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device_index,
    )


def score_sentences(
    sentences: Iterable[str],
    classifier,
    batch_size: int,
) -> List[Tuple[str, float]]:
    # The classifier can batch internally; we still request a batch_size for memory control.
    outputs = classifier(list(sentences), batch_size=batch_size, truncation=True)
    return [(out["label"], float(out["score"])) for out in outputs]


def process_file(
    input_path: Path,
    classifier,
    text_column: str,
    batch_size: int,
    output_dir: Path | None,
    suffix: str,
    overwrite: bool,
) -> Path:
    df = pd.read_csv(input_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {input_path}.")

    # Ensure strings for model input; keep track of rows we actually score.
    mask = df[text_column].notna()
    texts = df.loc[mask, text_column].astype(str)

    if len(texts) == 0:
        df["sentiment_label"] = None
        df["sentiment_score"] = None
    else:
        labels_scores = score_sentences(texts, classifier, batch_size)
        labels, scores = zip(*labels_scores)
        df["sentiment_label"] = None
        df["sentiment_score"] = None
        df.loc[mask, "sentiment_label"] = labels
        df.loc[mask, "sentiment_score"] = scores

    output_path = resolve_output_path(input_path, output_dir, suffix, overwrite)
    df.to_csv(output_path, index=False)
    return output_path


def main():
    args = parse_args()
    classifier = build_classifier(args.model, args.device)

    for path_str in args.inputs:
        input_path = Path(path_str)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path = process_file(
            input_path=input_path,
            classifier=classifier,
            text_column=args.text_column,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            suffix=args.suffix,
            overwrite=args.overwrite,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
