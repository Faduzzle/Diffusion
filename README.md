# Time Series Diffusion Framework

A modular framework for experimenting with diffusion models on time series data. Whether you're predicting Bitcoin prices, modeling weather patterns, or generating synthetic sensor data, this framework makes it easy to experiment with different approaches.

## Features

- **Modular Design**: Swap components (architectures, noise types, preprocessing methods) with a single config change
- **Built for Experimentation**: Compare different approaches systematically
- **Production Ready**: Includes training, evaluation, and inference pipelines
- **Extensible**: Easy to add new components and methods
- **Well-Documented**: Comprehensive documentation for both users and developers

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/time-series-diffusion.git
cd time-series-diffusion

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

1. **Prepare your data:**

```python
import pandas as pd
from src.data import prepare_dataset

# Load your time series data
data = pd.read_csv("your_data.csv")
train_data, val_data, test_data = prepare_dataset(
    data["value"].values,
    train_ratio=0.8,
    val_ratio=0.1
)
```

2. **Train a model:**

```bash
# Using default configuration
python -m src.train --data-path your_data.csv

# Using custom configuration
python -m src.train --config configs/your_config.yaml
```

3. **Generate predictions:**

```python
from src import DiffusionPredictor

predictor = DiffusionPredictor.from_checkpoint("checkpoints/best_model.pt")
predictions = predictor.predict(
    history=historical_data,
    num_samples=100,
    horizon=30
)
```

## Examples

### Bitcoin Price Prediction

```bash
# Use the pre-configured Bitcoin example
python -m src.train --config examples/bitcoin/config.yaml

# Evaluate the model
python -m src.evaluate \
    --checkpoint checkpoints/bitcoin_model.pt \
    --test-data data/bitcoin_test.csv
```

### Custom Time Series

```yaml
# Create a config file: configs/my_experiment.yaml
experiment:
  name: "my_time_series_experiment"
  
components:
  preprocessing:
    type: "standard"  # or "wavelet", "fourier", etc.
    
  architecture:
    type: "transformer"  # or "cnn", "rnn", etc.
    params:
      model_dim: 128
      num_layers: 4
      
data:
  history_len: 100  # How much history to use
  predict_len: 10   # How far to predict
  
training:
  batch_size: 32
  num_epochs: 50
```

## Key Concepts

### 1. Modular Components

The framework is built around swappable components:

- **Preprocessing**: How to prepare your data (normalization, transformations)
- **Architecture**: The neural network model (Transformer, CNN, RNN)
- **Noise**: How noise is added during training (standard, correlated, adaptive)
- **SDE**: The stochastic process governing diffusion (VP-SDE, VE-SDE)
- **Objective**: The training loss function (score matching, denoising)
- **Sampler**: How to generate samples (Euler, Heun, DPM-Solver)

### 2. Experiment Configuration

All experiments are defined through YAML configuration files. This makes it easy to:
- Track what settings produced which results
- Share experiments with others
- Run systematic comparisons

### 3. Built-in Evaluation

The framework includes comprehensive evaluation metrics:
- Prediction accuracy (MSE, MAE)
- Uncertainty quantification
- Distribution matching
- Domain-specific metrics (e.g., Sharpe ratio for financial data)

## Advanced Usage

### Comparing Multiple Approaches

```bash
# Compare different architectures
python -m src.compare \
    --configs configs/transformer.yaml configs/cnn.yaml configs/rnn.yaml \
    --output-dir comparisons/architectures
```

### Hyperparameter Search

```bash
# Run hyperparameter optimization
python -m src.search \
    --base-config configs/base.yaml \
    --search-space configs/search_space.yaml \
    --n-trials 100
```

### Custom Components

Create your own components by inheriting from base classes:

```python
from src.core import BaseArchitecture, Registry

@Registry.register("architecture", "my_custom_model")
class MyCustomModel(BaseArchitecture):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        # Your model definition
    
    def forward(self, x, t, condition=None):
        # Your forward pass
        return output
```

## Visualization

The framework includes built-in visualization tools:

```python
from src.visualization import plot_predictions

# Plot predictions with uncertainty bands
plot_predictions(
    true_values=test_data,
    predictions=predictions,
    save_path="results/predictions.png"
)
```

## Best Practices

1. **Start Simple**: Begin with standard components and default configurations
2. **Validate on Synthetic Data**: Test your setup on known patterns first
3. **Monitor Training**: Use TensorBoard or Weights & Biases integration
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Uncertainty Matters**: Always generate multiple samples to assess uncertainty

## Documentation

- **For AI/ML practitioners**: See [CLAUDE.md](CLAUDE.md) for technical details
- **API Reference**: Full API documentation at [docs/api/](docs/api/)
- **Tutorials**: Step-by-step guides in [docs/tutorials/](docs/tutorials/)
- **Papers**: Theoretical background in [docs/papers/](docs/papers/)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{time_series_diffusion,
  title = {Time Series Diffusion Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/time-series-diffusion}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This framework builds on ideas from:
- Score-based diffusion models (Song et al.)
- Denoising diffusion probabilistic models (Ho et al.)
- Transformer architectures for time series (Zhou et al.)

Special thanks to the open-source community for inspiration and contributions.