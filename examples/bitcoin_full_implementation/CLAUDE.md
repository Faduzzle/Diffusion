# Bitcoin Time Series Diffusion Model

## Overview

This project implements a state-of-the-art diffusion model for Bitcoin price prediction using a Transformer-based architecture. The model learns to generate realistic future price trajectories by modeling the stochastic differential equations (SDEs) that govern the diffusion process.

## Project Structure

```
bitcoin_diffusion/
├── src/                      # Source code
│   ├── models/              # Model implementations
│   │   ├── diffusion_model.py  # Main Transformer-based diffusion model
│   │   └── sde.py              # Stochastic Differential Equations
│   ├── data/                # Data processing
│   │   ├── dataset.py          # PyTorch dataset classes
│   │   └── preprocessing.py    # Data loading and normalization
│   ├── training/            # Training logic
│   │   └── trainer.py          # Training loop and optimization
│   ├── evaluation/          # Model evaluation
│   │   ├── predictor.py        # Generate predictions
│   │   └── metrics.py          # Evaluation metrics
│   └── utils/               # Utilities
│       ├── visualization.py    # Plotting functions
│       └── config.py           # Configuration management
├── configs/                 # Configuration files
│   └── bitcoin_default.yaml    # Default training config
├── data/                    # Data storage
│   ├── raw/                    # Raw Bitcoin price data
│   └── processed/              # Preprocessed tensors
├── scripts/                 # Executable scripts
│   ├── train.py                # Train the model
│   ├── predict.py              # Generate predictions
│   └── evaluate.py             # Evaluate model performance
└── notebooks/               # Jupyter notebooks
    └── analysis.ipynb          # Result analysis
```

## Architecture

### Diffusion Model

The core model (`DiffusionTransformer`) uses an encoder-decoder Transformer architecture:

1. **Encoder**: Processes historical price data to capture market context
2. **Decoder**: Predicts the score function for the reverse diffusion process
3. **Time Embedding**: Embeds the diffusion timestep into the model
4. **Positional Encoding**: Provides temporal awareness using sinusoidal encodings

Key features:
- Supports classifier-free guidance for controlled generation
- Optional latent diffusion for high-dimensional data
- Configurable architecture (layers, heads, dimensions)

### Stochastic Differential Equations

The model supports multiple SDE types:

1. **VP-SDE** (Variance Preserving): Default choice, maintains data variance
2. **VE-SDE** (Variance Exploding): Alternative with unbounded variance
3. **Sub-VP-SDE**: Improved numerical stability for discrete sampling

The forward process gradually adds noise: `dx = f(x,t)dt + g(t)dW`
The reverse process removes noise using the learned score function.

### Classifier-Free Guidance

The model implements classifier-free guidance (CFG) for controllable generation:

**Training**: During training, historical conditioning is randomly dropped with probability `cond_drop_prob`, teaching the model both conditional and unconditional generation.

**Inference**: At inference time, predictions are computed using:
```
score = uncond_score + guidance_scale * (cond_score - uncond_score)
```

**Guidance Scale Effects**:
- `guidance_scale = 1.0`: Standard conditional generation
- `guidance_scale > 1.0`: Stronger conditioning on historical patterns
- `guidance_scale < 1.0`: Weaker conditioning, more unconditional behavior

### Data Pipeline

The data pipeline handles:
- CSV loading with automatic date parsing
- Log return computation for stationarity
- Multiple normalization methods (z-score, min-max, robust)
- Sliding window dataset creation
- Train/validation/test splitting

## Key Algorithms

### Training Process

1. **Forward Diffusion**: Add noise to future price trajectories according to the SDE
2. **Score Matching**: Train the model to predict the score (gradient of log probability)
3. **Loss Function**: Weighted MSE between predicted and true scores
4. **Optimization**: AdamW with cosine annealing learning rate schedule

### Generation Process

1. **Sample from Prior**: Start with Gaussian noise at t=1
2. **Reverse Diffusion**: Iteratively denoise using the learned score function with CFG
3. **Classifier-Free Guidance**: Combine conditional and unconditional scores for controllable generation
4. **Numerical Integration**: Use Euler-Maruyama or higher-order solvers
5. **Ensemble Predictions**: Generate multiple trajectories for uncertainty quantification

## Usage Examples

### Training a Model

```python
from src.training import Trainer
from src.utils.config import load_config

# Load configuration
config = load_config('configs/bitcoin_default.yaml')

# Initialize trainer
trainer = Trainer(config)

# Train model
trainer.train()
```

### Generating Predictions

```python
from src.evaluation import DiffusionPredictor

# Load trained model
predictor = DiffusionPredictor.from_checkpoint('models/best_model.pt')

# Generate predictions with classifier-free guidance
predictions = predictor.predict(
    history=historical_prices,
    num_samples=100,
    num_steps=1000,
    guidance_scale=2.0  # Control conditioning strength
)
```

### Data Preprocessing

```python
from src.data import prepare_bitcoin_data

# Prepare data for training
data_paths = prepare_bitcoin_data(
    raw_data_path='data/raw/bitcoin_prices.csv',
    output_dir='data/processed/',
    config={'norm_method': 'standard'}
)
```

## Configuration

The model is configured via YAML files. Key parameters:

```yaml
model:
  type: "DiffusionTransformer"
  input_dim: 1              # Univariate time series
  model_dim: 256            # Hidden dimension
  num_heads: 8              # Attention heads
  num_layers: 6             # Transformer layers
  history_len: 252          # ~1 year of trading days
  predict_len: 21           # ~1 month prediction

sde:
  type: "vpsde"
  beta_min: 0.1
  beta_max: 20.0

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  gradient_clip: 1.0
  ema_decay: 0.999

data:
  norm_method: "standard"    # z-score normalization
  returns_method: "log"      # log returns
  train_ratio: 0.8
  val_ratio: 0.1
```

## Experiments

### Baseline Comparisons

The model is compared against:
1. **ARIMA**: Classical time series model
2. **GARCH**: Volatility modeling
3. **Random Walk**: Naive baseline
4. **Historical Mean**: Simple average

### Evaluation Metrics

- **MSE**: Mean squared error of predictions
- **MAE**: Mean absolute error
- **Sharpe Ratio**: Risk-adjusted returns
- **Directional Accuracy**: Correct prediction of price direction
- **Quantile Coverage**: Calibration of uncertainty estimates

### Key Findings

1. Diffusion models capture complex dependencies better than classical methods
2. Transformer architecture effectively uses historical context
3. Classifier-free guidance improves prediction stability
4. Ensemble predictions provide well-calibrated uncertainty estimates

## Best Practices

### Data Preparation
- Always use log returns for financial data
- Normalize using training statistics only
- Ensure no data leakage between train/test sets

### Model Training
- Use gradient clipping to prevent instability
- Monitor validation loss for early stopping
- Save checkpoints regularly
- Use EMA for stable inference

### Prediction
- Generate multiple samples for uncertainty
- Use appropriate number of diffusion steps
- Consider market regime when interpreting results
- Validate predictions on out-of-sample data

## Common Commands

```bash
# Train model with CFG
python scripts/train.py --config configs/bitcoin_cfg_experiment.yaml

# Generate predictions with different guidance scales
python scripts/predict.py --checkpoint models/best_model.pt --num-samples 100

# Run CFG experiment
python scripts/run_cfg_experiment.py --checkpoint models/best_model.pt --data data/test.csv

# Evaluate performance
python scripts/evaluate.py --predictions outputs/predictions.csv --ground-truth data/test.csv
```

## Troubleshooting

### NaN Loss During Training
- Reduce learning rate
- Increase gradient clipping
- Check data normalization

### Poor Predictions
- Increase model capacity
- Tune SDE parameters
- Add more historical context
- Check for data quality issues

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Limit dataset size with max_samples
- Enable mixed precision training

## Future Improvements

1. **Multi-asset modeling**: Extend to portfolio of cryptocurrencies
2. **Conditional generation**: Condition on market indicators
3. **Adaptive noise schedules**: Learn optimal SDE parameters
4. **Real-time updates**: Online learning capabilities
5. **Interpretability**: Attention visualization and feature importance