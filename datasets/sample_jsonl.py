#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

def sample_jsonl(input_file: Path, output_file: Path, sample_size: int) -> None:
    """
    Randomly sample lines from a JSONL file.
    
    Args:
        input_file (Path): Path to input JSONL file
        output_file (Path): Path to output JSONL file
        sample_size (int): Number of lines to sample
    """
    # Read all lines into memory
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if we have enough lines
    total_lines = len(lines)
    if total_lines < sample_size:
        raise ValueError(f"Input file has only {total_lines} lines, cannot sample {sample_size} lines")
    
    # Randomly sample lines
    sampled_lines = random.sample(lines, sample_size)
    
    # Write sampled lines to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line)
            
    print(f"Successfully sampled {sample_size} lines from {total_lines} total lines")

def main():
    parser = argparse.ArgumentParser(description='Randomly sample lines from a JSONL file')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('output_file', help='Output JSONL file')
    parser.add_argument('--sample-size', type=int, default=250,
                      help='Number of lines to sample (default: 250)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    try:
        sample_jsonl(Path(args.input_file), Path(args.output_file), args.sample_size)
    except Exception as e:
        print(f"Error during sampling: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()