#!/usr/bin/env python3
"""
Convert parquet data to Unsloth SFT JSONL format.

Usage:
    python convert_parquet_to_unsloth.py \
        --input data/train-00000-of-00001.parquet \
        --output data/train_unsloth.jsonl

    # Also supports .parquet.gz compressed files
    python convert_parquet_to_unsloth.py \
        --input data/train-00000-of-00001.parquet.gz \
        --output data/train_unsloth.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def convert_parquet_to_jsonl(
    input_path: str,
    output_path: str,
    compression: str = "default",
) -> int:
    """
    Convert a parquet file to Unsloth SFT JSONL format.

    Args:
        input_path: Path to the input parquet file
        output_path: Path to the output JSONL file
        compression: Compression level for pandas (default, infer, or integer)

    Returns:
        Number of records written
    """
    # Read the parquet file
    df = pd.read_parquet(input_path)

    print(f"Input: {input_path}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()}")

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each row to JSONL format
    records = []
    for idx, row in df.iterrows():
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": row["problem"],
                },
                {
                    "role": "assistant",
                    "content": f"<thinking>\n{row['thinking']}\n</thinking>\n\n{row['solution']}",
                },
            ],
            "metadata": {
                "id": str(row["id"]),
                "difficulty": row["difficulty"],
                "category": row["category"],
            },
        }
        records.append(record)

    # Write to JSONL file
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nOutput: {output_path}")
    print(f"  Records written: {len(records)}")

    # Print category distribution
    cat_counts = df["category"].value_counts().head(10)
    print(f"\nCategory distribution:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")

    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet data to Unsloth SFT JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/train-00000-of-00001.parquet --output data/train_unsloth.jsonl
  %(prog)s -i data/train.parquet.gz -o data/train_unsloth.jsonl
        """,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input parquet file (.parquet or .parquet.gz)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to the output JSONL file",
    )
    parser.add_argument(
        "--compression",
        default="default",
        choices=["default", "infer", "snappy", "gzip", "bz2", "lz4", "zstd"],
        help="Compression level for pandas (default: %(default)s)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        count = convert_parquet_to_jsonl(
            args.input,
            args.output,
            compression=args.compression,
        )
        print(f"\n✓ Conversion complete: {count} records written to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
