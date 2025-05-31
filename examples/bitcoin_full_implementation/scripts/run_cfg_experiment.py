#!/usr/bin/env python3
"""
Run classifier-free guidance experiments with different guidance scales.

Usage:
    python run_cfg_experiment.py --checkpoint models/best_model.pt --data data/test.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import DiffusionPredictor, MetricCalculator
from src.data import DataPreprocessor
from src.utils import load_config


def run_cfg_experiment(
    checkpoint_path: str,
    data_path: str,
    output_dir: str = "outputs/cfg_experiment",
    guidance_scales: list = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
    num_samples: int = 100,
    num_steps: int = 1000
):
    """Run classifier-free guidance experiment."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictor
    print("Loading model...")
    predictor = DiffusionPredictor.from_checkpoint(checkpoint_path)
    
    # Load data
    print("Loading data...")
    if data_path.endswith('.pt'):
        # Preprocessed data
        preprocessor = DataPreprocessor({})
        data, metadata = preprocessor.load_processed_data(data_path)
        predictor.preprocessor = preprocessor
        predictor.preprocessor.normalization_params = metadata.get('normalization_params', {})
    else:
        # Raw CSV data
        df = pd.read_csv(data_path)
        price_col = 'Close' if 'Close' in df.columns else df.columns[-1]
        prices = df[price_col].values
        
        # Compute returns
        preprocessor = DataPreprocessor({})
        returns = preprocessor.compute_returns(prices, method='log')
        data = preprocessor.normalize(
            torch.tensor(returns, dtype=torch.float32).unsqueeze(-1),
            method='standard'
        )
        predictor.preprocessor = preprocessor
    
    # Extract sample
    history_len = predictor.model.history_len
    predict_len = predictor.model.predict_len
    
    start_idx = len(data) // 2
    sample_history = data[start_idx:start_idx + history_len].unsqueeze(0)
    sample_future = data[start_idx + history_len:start_idx + history_len + predict_len]
    
    print(f"Using sample starting at index {start_idx}")
    print(f"History shape: {sample_history.shape}")
    print(f"Future shape: {sample_future.shape}")
    
    # Run experiments
    results = []
    predictions = {}
    
    for scale in guidance_scales:
        print(f"\nGenerating predictions with guidance scale {scale}...")
        
        pred = predictor.predict(
            history=sample_history,
            num_samples=num_samples,
            num_steps=num_steps,
            guidance_scale=scale,
            denormalize=True,
            return_dict=True
        )
        
        predictions[scale] = pred
        
        # Compute metrics if ground truth available
        if len(sample_future) == predict_len:
            if predictor.preprocessor is not None:
                future_denorm = predictor.preprocessor.denormalize(sample_future).numpy()
            else:
                future_denorm = sample_future.numpy()
            
            metric_calc = MetricCalculator()
            metrics = metric_calc.compute_all_metrics(pred, future_denorm)
            
            result = {'guidance_scale': scale}
            result.update(metrics)
            results.append(result)
            
            print(f"  MSE: {metrics.get('mse', 0):.4f}")
            print(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f}")
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'cfg_experiment_results.csv', index=False)
        print(f"\nResults saved to {output_dir / 'cfg_experiment_results.csv'}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Plot predictions comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Prepare data for plotting
    if predictor.preprocessor is not None:
        history_denorm = predictor.preprocessor.denormalize(sample_history.squeeze()).numpy()
        if len(sample_future) == predict_len:
            future_denorm = predictor.preprocessor.denormalize(sample_future).numpy()
        else:
            future_denorm = None
    else:
        history_denorm = sample_history.squeeze().numpy()
        future_denorm = sample_future.numpy() if len(sample_future) == predict_len else None
    
    for i, scale in enumerate(guidance_scales):
        ax = axes[i]
        pred = predictions[scale]
        
        # Time indices
        history_time = np.arange(-history_len, 0)
        future_time = np.arange(predict_len)
        
        # Plot history
        ax.plot(history_time, history_denorm, 'k-', label='History', linewidth=2, alpha=0.7)
        
        # Plot mean prediction
        ax.plot(future_time, pred['mean'][0], 'b-', label='Mean Prediction', linewidth=2)
        
        # Plot confidence intervals
        if 'quantiles' in pred:
            ax.fill_between(
                future_time,
                pred['quantiles'][0.1][0],
                pred['quantiles'][0.9][0],
                alpha=0.2, color='blue', label='80% CI'
            )
        
        # Plot ground truth if available
        if future_denorm is not None:
            ax.plot(future_time, future_denorm, 'r-', label='Ground Truth', linewidth=2)
        
        # Formatting
        ax.set_title(f'Guidance Scale = {scale}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Log Returns')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.suptitle('Classifier-Free Guidance Experiment Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_dir / 'cfg_predictions_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot metrics if available
    if results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # MSE
        axes[0].plot(results_df['guidance_scale'], results_df['mse'], 'o-', linewidth=2)
        axes[0].set_xlabel('Guidance Scale')
        axes[0].set_ylabel('MSE')
        axes[0].set_title('Mean Squared Error')
        axes[0].grid(True, alpha=0.3)
        
        # Directional Accuracy
        axes[1].plot(results_df['guidance_scale'], results_df['directional_accuracy'], 
                    'o-', linewidth=2, color='orange')
        axes[1].set_xlabel('Guidance Scale')
        axes[1].set_ylabel('Directional Accuracy')
        axes[1].set_title('Directional Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # Uncertainty (std)
        uncertainties = [predictions[scale]['std'][0].mean() for scale in guidance_scales]
        axes[2].plot(guidance_scales, uncertainties, 'o-', linewidth=2, color='green')
        axes[2].set_xlabel('Guidance Scale')
        axes[2].set_ylabel('Mean Prediction Std')
        axes[2].set_title('Prediction Uncertainty')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cfg_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Find optimal scales
        optimal_mse_scale = results_df.loc[results_df['mse'].idxmin(), 'guidance_scale']
        optimal_dir_scale = results_df.loc[results_df['directional_accuracy'].idxmax(), 'guidance_scale']
        
        print(f"\nOptimal guidance scales:")
        print(f"  Best MSE: {optimal_mse_scale}")
        print(f"  Best Directional Accuracy: {optimal_dir_scale}")
    
    print(f"\nExperiment completed! Results saved to {output_dir}")
    return predictions, results


def main():
    parser = argparse.ArgumentParser(description='Run CFG experiment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='Test data path')
    parser.add_argument('--output', type=str, default='outputs/cfg_experiment', help='Output directory')
    parser.add_argument('--scales', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                       help='Guidance scales to test')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of prediction samples')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of diffusion steps')
    args = parser.parse_args()
    
    predictions, results = run_cfg_experiment(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_dir=args.output,
        guidance_scales=args.scales,
        num_samples=args.num_samples,
        num_steps=args.num_steps
    )


if __name__ == '__main__':
    main()