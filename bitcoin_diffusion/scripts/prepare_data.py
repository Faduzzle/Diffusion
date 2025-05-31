#!/usr/bin/env python3
"""
Prepare Bitcoin data for training.

Usage:
    python prepare_data.py --input data/raw/bitcoin.csv --output data/processed/
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import prepare_bitcoin_data
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description='Prepare Bitcoin data')
    parser.add_argument('--input', type=str, required=True, help='Raw data CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--config', type=str, help='Config file for preprocessing options')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation data ratio')
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        preprocess_config = config.get('data', {})
    else:
        preprocess_config = {
            'norm_method': 'standard',
            'returns_method': 'log',
            'fill_method': 'ffill'
        }
    
    # Prepare data
    print("Preparing Bitcoin data...")
    output_paths = prepare_bitcoin_data(
        raw_data_path=args.input,
        output_dir=args.output,
        config=preprocess_config,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    print("\nData prepared successfully!")
    print("Output files:")
    for split, path in output_paths.items():
        print(f"  {split}: {path}")
    
    # Load and print statistics
    from src.data import DataPreprocessor
    preprocessor = DataPreprocessor(preprocess_config)
    
    for split, path in output_paths.items():
        data, metadata = preprocessor.load_processed_data(path)
        print(f"\n{split.capitalize()} data shape: {data.shape}")
        print(f"  Mean: {data.mean().item():.4f}")
        print(f"  Std: {data.std().item():.4f}")
        print(f"  Min: {data.min().item():.4f}")
        print(f"  Max: {data.max().item():.4f}")


if __name__ == '__main__':
    main()