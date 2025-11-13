#!/usr/bin/env python3
"""
Script to randomly split a JSONL dataset into two parts.
"""

import json
import random
import argparse
from pathlib import Path


def split_jsonl(input_file, output_file1, output_file2, split_ratio=0.5, seed=42):
    """
    Randomly split a JSONL file into two parts.

    Args:
        input_file: Path to input JSONL file
        output_file1: Path to first output JSONL file
        output_file2: Path to second output JSONL file
        split_ratio: Ratio for first file (default 0.5 for 50/50 split)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Read all lines
    print(f"Reading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Total lines: {total_lines}")

    # Shuffle the lines randomly
    print("Shuffling data...")
    random.shuffle(lines)

    # Calculate split point
    split_point = int(total_lines * split_ratio)
    part1 = lines[:split_point]
    part2 = lines[split_point:]

    # Write first part
    print(f"Writing {len(part1)} lines to {output_file1}...")
    with open(output_file1, 'w', encoding='utf-8') as f:
        f.writelines(part1)

    # Write second part
    print(f"Writing {len(part2)} lines to {output_file2}...")
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.writelines(part2)

    print("\nSplit complete!")
    print(f"  Part 1: {len(part1)} lines ({len(part1)/total_lines*100:.1f}%)")
    print(f"  Part 2: {len(part2)} lines ({len(part2)/total_lines*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Randomly split a JSONL dataset into two parts'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output1',
        type=str,
        default=None,
        help='Path to first output file (default: input_file_part1.jsonl)'
    )
    parser.add_argument(
        '--output2',
        type=str,
        default=None,
        help='Path to second output file (default: input_file_part2.jsonl)'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=0.5,
        help='Split ratio for first file (default: 0.5 for 50/50 split)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Generate default output filenames if not provided
    input_path = Path(args.input_file)
    if args.output1 is None:
        args.output1 = str(input_path.parent / f"{input_path.stem}_part1{input_path.suffix}")
    if args.output2 is None:
        args.output2 = str(input_path.parent / f"{input_path.stem}_part2{input_path.suffix}")

    split_jsonl(
        args.input_file,
        args.output1,
        args.output2,
        args.ratio,
        args.seed
    )


if __name__ == '__main__':
    main()
