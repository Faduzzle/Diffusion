# Bitcoin Time Series Diffusion Model

A clean, modular implementation of diffusion models for Bitcoin price prediction using Transformer architectures.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Prepare Data

```bash
python scripts/prepare_data.py \
    --input data/raw/bitcoin.csv \
    --output data/processed/
```

### Train Model

```bash
python scripts/train.py --config configs/bitcoin_default.yaml
```

### Generate Predictions

```bash
python scripts/predict.py \
    --checkpoint models/checkpoints/best_model.pt \
    --data data/raw/bitcoin.csv \
    --plot
```

## Project Structure

- `src/` - Core source code
  - `models/` - Diffusion model and SDE implementations
  - `data/` - Data loading and preprocessing
  - `training/` - Training logic
  - `evaluation/` - Prediction and metrics
  - `utils/` - Configuration and visualization
- `configs/` - YAML configuration files
- `scripts/` - Executable scripts
- `data/` - Data storage
- `models/` - Saved model checkpoints

## Key Features

- **Transformer-based Architecture**: Encoder-decoder design for temporal modeling
- **Multiple SDE Types**: VP-SDE, VE-SDE, and Sub-VP-SDE
- **Flexible Data Pipeline**: Handles various normalization and preprocessing options
- **Comprehensive Evaluation**: Multiple metrics for probabilistic predictions
- **Clean Code**: Modular design with type hints and documentation

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation on the architecture, algorithms, and usage.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bitcoin_diffusion,
  title = {Bitcoin Time Series Diffusion Model},
  year = {2024},
  url = {https://github.com/yourusername/bitcoin-diffusion}
}
```