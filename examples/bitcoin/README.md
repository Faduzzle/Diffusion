# Bitcoin Price Diffusion Example

This example demonstrates how to use the time series diffusion framework for Bitcoin price prediction.

## Overview

This example:
- Uses 1 year of historical Bitcoin prices to predict 1 month ahead
- Applies log returns preprocessing for stationarity
- Uses a Transformer architecture with 6 layers
- Implements classifier-free guidance for better predictions

## Quick Start

### 1. Prepare Data

First, ensure you have Bitcoin price data in CSV format with columns:
- `Date`: Date column
- `Close`: Closing price

You can download Bitcoin data from:
- Yahoo Finance
- CoinGecko API
- Your preferred data provider

### 2. Update Configuration

Edit `config.yaml` to point to your data:

```yaml
data:
  data_path: path/to/your/bitcoin_data.csv
```

### 3. Run Training

```bash
# From the repository root
python -m src.train --config examples/bitcoin/config.yaml
```

### 4. Evaluate Model

```bash
# Evaluate on test set
python -m src.evaluate \
    --checkpoint experiments/bitcoin_diffusion_*/checkpoints/best_model.pt \
    --config examples/bitcoin/config.yaml
```

### 5. Generate Predictions

```python
from src import DiffusionPredictor
import pandas as pd

# Load model
predictor = DiffusionPredictor.from_checkpoint(
    "experiments/bitcoin_diffusion_*/checkpoints/best_model.pt"
)

# Load recent data
data = pd.read_csv("bitcoin_data.csv")
recent_prices = data['Close'].values[-252:]  # Last year

# Generate predictions
predictions = predictor.predict(
    history=recent_prices,
    num_samples=1000,  # Generate 1000 possible futures
    horizon=21  # Predict 21 days ahead
)

# Get statistics
mean_prediction = predictions.mean(axis=0)
lower_bound = predictions.quantile(0.1, axis=0)
upper_bound = predictions.quantile(0.9, axis=0)
```

## Configuration Details

### Data Preprocessing

The example uses log returns preprocessing:
```yaml
preprocessing:
  type: log_returns
  params:
    normalize: true  # Standardize returns
```

This converts prices to log returns: `log(p_t / p_{t-1})`

### Model Architecture

The Transformer architecture is configured for financial time series:
```yaml
architecture:
  type: transformer
  params:
    model_dim: 256      # Hidden dimension
    num_heads: 8        # Attention heads
    num_layers: 6       # Transformer blocks
    cond_drop_prob: 0.1 # For classifier-free guidance
```

### Training Settings

Optimized for Bitcoin's volatility:
```yaml
training:
  batch_size: 32
  learning_rate: 1e-4
  gradient_clip: 1.0  # Important for stability
  use_ema: true       # Exponential moving average
```

### Evaluation Metrics

Financial-specific metrics included:
```yaml
evaluation:
  metrics:
    - sharpe_ratio         # Risk-adjusted returns
    - directional_accuracy # Prediction direction accuracy
    - quantile_coverage    # Uncertainty calibration
```

## Advanced Usage

### 1. Hyperparameter Tuning

```bash
# Run hyperparameter search
python -m src.search \
    --base-config examples/bitcoin/config.yaml \
    --search-config examples/bitcoin/search_space.yaml \
    --n-trials 50
```

### 2. Multi-GPU Training

```yaml
# In config.yaml
training:
  distributed: true
  num_gpus: 2
```

### 3. Custom Preprocessing

Create a custom preprocessor for technical indicators:

```python
from src import BasePreprocessor, Registry
import talib

@Registry.register("preprocessing", "technical_indicators")
class TechnicalPreprocessor(BasePreprocessor):
    def transform(self, data):
        # Add RSI, MACD, etc.
        features = {
            'price': data,
            'rsi': talib.RSI(data),
            'macd': talib.MACD(data)[0]
        }
        return np.stack(list(features.values()), axis=-1)
```

### 4. Backtesting

Run trading simulation:

```python
from src.evaluation import Backtester

backtester = Backtester(
    initial_capital=10000,
    transaction_cost=0.001
)

results = backtester.run(
    predictions=predictions,
    actual_prices=test_prices,
    strategy="long_only"  # or "long_short"
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Experiment Variations

### 1. Different Time Horizons

```yaml
# Short-term (1 week)
data:
  history_len: 60   # ~2 months
  predict_len: 7    # 1 week

# Long-term (3 months)
data:
  history_len: 504  # ~2 years
  predict_len: 63   # ~3 months
```

### 2. Alternative Architectures

```yaml
# Try CNN for faster inference
architecture:
  type: cnn
  params:
    channels: [64, 128, 256, 512]
    kernel_sizes: [7, 5, 3, 3]
```

### 3. Different SDEs

```yaml
# Try VE-SDE
sde:
  type: vesde
  params:
    sigma_min: 0.01
    sigma_max: 50.0
```

## Results Interpretation

### Understanding Predictions

The model generates multiple possible future trajectories. This captures:
- **Uncertainty**: Wider prediction bands = higher uncertainty
- **Multi-modality**: Multiple possible scenarios
- **Tail risks**: Extreme events in the distribution

### Key Metrics

1. **MSE/MAE**: Point prediction accuracy
2. **CRPS**: Quality of probabilistic predictions
3. **Sharpe Ratio**: Risk-adjusted performance
4. **Directional Accuracy**: Correct up/down predictions

### Visualization

The framework generates several plots:
- `predictions.png`: Sample trajectories with confidence bands
- `metrics.png`: Performance metrics over time
- `attention.png`: What historical periods the model focuses on

## Troubleshooting

### Common Issues

1. **Unstable Training**
   - Reduce learning rate
   - Increase gradient clipping
   - Use smaller beta_max in SDE

2. **Poor Predictions**
   - Increase model size
   - Try different preprocessing
   - Adjust history/predict lengths

3. **Overfitting**
   - Add dropout
   - Reduce model size
   - Use more data augmentation

## References

- Original Bitcoin diffusion implementation
- [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)