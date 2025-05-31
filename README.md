# Bitcoin Time Series Diffusion with Classifier-Free Guidance

A clean, modular implementation of diffusion models for Bitcoin price prediction using Transformer architectures with classifier-free guidance.

## Quick Start

### 1. Prepare Data
```bash
python prepare_data.py
```

### 2. Train Model

**Standard Training:**
```bash
python train_bitcoin.py
```

**With Classifier-Free Guidance:**
```bash
python train_bitcoin.py --cfg-experiment
```

### 3. Run CFG Experiments
```bash
cd bitcoin_diffusion
python scripts/run_cfg_experiment.py \
    --checkpoint models/checkpoints_cfg/best_model.pt \
    --data data/processed/bitcoin_test.pt
```

## Classifier-Free Guidance

This implementation includes state-of-the-art classifier-free guidance (CFG) for controllable time series generation:

- **guidance_scale = 1.0**: Standard conditional generation
- **guidance_scale > 1.0**: Stronger conditioning on historical data
- **guidance_scale < 1.0**: More unconditional, diverse generation

### Interactive Demo
Open `bitcoin_diffusion/notebooks/classifier_free_guidance_demo.ipynb` to explore CFG effects interactively.

## Project Structure

```
bitcoin_diffusion/
├── src/                    # Core source code
│   ├── models/            # Diffusion model with CFG
│   ├── data/              # Data loading and preprocessing  
│   ├── training/          # Training with CFG support
│   ├── evaluation/        # Prediction with guidance control
│   └── utils/             # Configuration and visualization
├── configs/               # YAML configuration files
├── scripts/               # Executable scripts
├── notebooks/             # Interactive demos
└── data/                  # Data storage
```

## Key Features

- **Transformer-based Architecture**: Encoder-decoder design for temporal modeling
- **Classifier-Free Guidance**: Control conditioning strength during generation
- **Multiple SDE Types**: VP-SDE, VE-SDE, and Sub-VP-SDE
- **Flexible Data Pipeline**: Various normalization and preprocessing options
- **Comprehensive Evaluation**: Multiple metrics for probabilistic predictions
- **Clean Code**: Modular design with type hints and documentation

## Configuration

Models are configured via YAML files:

- `configs/bitcoin_default.yaml` - Standard training
- `configs/bitcoin_cfg_experiment.yaml` - CFG-optimized training

Key CFG parameters:
```yaml
model:
  cond_drop_prob: 0.15      # Conditioning dropout during training

evaluation:
  guidance_scale: 2.0       # Guidance strength (1.0 = no guidance)
```

## Documentation

See `bitcoin_diffusion/CLAUDE.md` for detailed documentation on architecture, algorithms, and usage.

## Citation

```bibtex
@software{bitcoin_diffusion_cfg,
  title = {Bitcoin Time Series Diffusion with Classifier-Free Guidance},
  year = {2024},
  url = {https://github.com/Faduzzle/Diffusion}
}
```