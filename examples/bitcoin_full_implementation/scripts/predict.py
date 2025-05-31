#!/usr/bin/env python3
"""
Generate predictions using trained diffusion model.

Usage:
    python predict.py --checkpoint models/best_model.pt --data data/test.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import DiffusionPredictor
from src.data import DataPreprocessor
from src.utils import plot_predictions


def main():
    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='Input data CSV path')
    parser.add_argument('--history-len', type=int, default=252, help='History window length')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of prediction samples')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--output', type=str, help='Output path for predictions')
    parser.add_argument('--plot', action='store_true', help='Plot predictions')
    args = parser.parse_args()
    
    # Load predictor
    print("Loading model...")
    predictor = DiffusionPredictor.from_checkpoint(args.checkpoint)
    
    # Load and preprocess data
    print("Loading data...")
    df = pd.read_csv(args.data)
    
    # Extract price column
    price_col = 'Close' if 'Close' in df.columns else df.columns[-1]
    prices = df[price_col].values
    
    # Compute returns
    preprocessor = DataPreprocessor({})
    returns = preprocessor.compute_returns(prices, method='log')
    
    # Normalize
    returns_tensor = preprocessor.normalize(
        torch.tensor(returns, dtype=torch.float32).unsqueeze(-1),
        method='standard'
    )
    
    # Extract history window
    history = returns_tensor[-args.history_len:].unsqueeze(0)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predictor.predict(
        history=history,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        denormalize=True,
        return_dict=True
    )
    
    # Save predictions
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy arrays
        np.savez(
            output_path,
            samples=predictions['samples'],
            mean=predictions['mean'],
            std=predictions['std'],
            **predictions['quantiles']
        )
        print(f"Predictions saved to {output_path}")
    
    # Plot if requested
    if args.plot:
        plot_predictions(
            predictions=predictions,
            history=returns_tensor[-args.history_len:].numpy(),
            title="Bitcoin Return Predictions"
        )
    
    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"Mean prediction shape: {predictions['mean'].shape}")
    print(f"First 5 mean predictions: {predictions['mean'][0, :5]}")
    print(f"Prediction std: {predictions['std'][0].mean():.4f}")


if __name__ == '__main__':
    main()