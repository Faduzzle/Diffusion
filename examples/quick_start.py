"""
Quick start example for Time Series Diffusion Framework.

This script demonstrates the basic usage of the framework with synthetic data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.experiments import ConfigBuilder, ExperimentRunner
from src.core import Registry


def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """Generate synthetic time series data for testing."""
    t = np.linspace(0, 4 * np.pi, n_samples)
    
    # Create a complex pattern: trend + seasonality + noise
    trend = 0.5 * t
    seasonal = 2 * np.sin(t) + np.sin(2 * t) + 0.5 * np.sin(4 * t)
    noise = noise_level * np.random.randn(n_samples)
    
    data = trend + seasonal + noise
    return data


def main():
    """Run a quick experiment with synthetic data."""
    
    print("Time Series Diffusion Framework - Quick Start")
    print("=" * 50)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000)
    
    # Save data
    data_dir = Path("data/synthetic")
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "synthetic_data.npy", data)
    
    # Plot data
    plt.figure(figsize=(12, 4))
    plt.plot(data[:200])
    plt.title("Synthetic Time Series (first 200 points)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(data_dir / "synthetic_data.png")
    plt.close()
    
    print(f"   Data shape: {data.shape}")
    print(f"   Data saved to: {data_dir}")
    
    # 2. Create experiment configuration
    print("\n2. Creating experiment configuration...")
    
    config = (ConfigBuilder("synthetic_quick_start")
             .description("Quick start example with synthetic data")
             .tags("synthetic", "example", "quick-start")
             
             # Components
             .component("preprocessing", "standard")
             .component("architecture", "transformer",
                       model_dim=128,
                       num_heads=4,
                       num_layers=3,
                       ff_dim=512)
             .component("sde", "vpsde",
                       beta_min=0.1,
                       beta_max=20.0)
             .component("objective", "score_matching")
             .component("sampler", "euler")
             
             # Data settings
             .data(
                 data_path=str(data_dir / "synthetic_data.npy"),
                 history_len=50,
                 predict_len=10,
                 train_ratio=0.7,
                 val_ratio=0.15,
                 test_ratio=0.15
             )
             
             # Training settings
             .training(
                 batch_size=16,
                 num_epochs=10,  # Quick training for demo
                 learning_rate=1e-3,
                 device="auto"
             )
             
             # Evaluation settings
             .evaluation(
                 num_samples=50,
                 num_steps=100,
                 metrics=["mse", "mae"]
             )
             
             .build())
    
    # Save config
    config_path = Path("configs/synthetic_quick_start.yaml")
    config_path.parent.mkdir(exist_ok=True)
    config.save(config_path)
    print(f"   Config saved to: {config_path}")
    
    # 3. Create and run experiment
    print("\n3. Running experiment...")
    print("   This is a minimal example - training for only 10 epochs")
    
    # Note: In a real scenario, you would implement the missing components
    print("\n   NOTE: This is a demonstration of the framework structure.")
    print("   To actually run this, you need to:")
    print("   1. Implement the missing components (objectives, samplers, etc.)")
    print("   2. Create the data loading utilities")
    print("   3. Implement the training and evaluation logic")
    
    # Show what would happen
    print("\n4. Framework workflow:")
    print("   a. Load and preprocess data using StandardPreprocessor")
    print("   b. Create TransformerArchitecture model with specified params")
    print("   c. Setup VP-SDE for the diffusion process")
    print("   d. Train using score matching objective")
    print("   e. Generate samples using Euler sampler")
    print("   f. Evaluate with MSE and MAE metrics")
    
    # 5. Show registry contents
    print("\n5. Available components in registry:")
    components = Registry.list()
    for comp_type, comp_names in components.items():
        print(f"   {comp_type}: {comp_names}")
    
    print("\n" + "=" * 50)
    print("Quick start complete!")
    print("\nNext steps:")
    print("1. Implement additional components as needed")
    print("2. Try with real time series data")
    print("3. Experiment with different architectures and settings")
    print("4. Use the comparison tools to find best configurations")


if __name__ == "__main__":
    # Import components to trigger registration
    try:
        from src.components import (
            architectures, preprocessing, sde
        )
    except ImportError:
        print("Note: Some imports failed - this is expected in the demo")
    
    main()