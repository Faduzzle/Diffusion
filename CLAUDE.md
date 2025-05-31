# Time Series Diffusion Framework - Technical Documentation for AI Models

## Overview

This is a modular framework for experimenting with diffusion models on time series data. The framework uses a plugin-based architecture that allows easy swapping of components and systematic comparison of different approaches.

## Architecture

### Core Components

1. **Registry System** (`src/core/registry.py`)
   - Central registry for all pluggable components
   - Automatic discovery and registration of components
   - Type-safe component retrieval

2. **Base Classes** (`src/core/base.py`)
   - `BasePreprocessor`: Interface for data preprocessing
   - `BaseArchitecture`: Interface for model architectures
   - `BaseNoise`: Interface for noise schedules
   - `BaseSDE`: Interface for stochastic differential equations
   - `BaseObjective`: Interface for training objectives
   - `BaseSampler`: Interface for sampling algorithms

3. **Experiment System** (`src/experiments/`)
   - Configuration management
   - Hyperparameter tracking
   - Results logging and comparison

### Component Types

#### Preprocessing (`src/components/preprocessing/`)
- `RawPreprocessor`: No preprocessing, just normalization
- `WaveletPreprocessor`: Wavelet decomposition
- `FourierPreprocessor`: Fourier transform features
- `TechnicalPreprocessor`: Technical indicators for financial data

#### Architectures (`src/components/architectures/`)
- `TransformerArchitecture`: Attention-based model (current default)
- `CNNArchitecture`: Convolutional neural network
- `RNNArchitecture`: Recurrent neural network variants
- `HybridArchitecture`: Combined architectures

#### Noise Types (`src/components/noise/`)
- `StandardNoise`: IID Gaussian noise
- `CorrelatedNoise`: Temporally correlated noise
- `StructuredNoise`: Noise with specific patterns
- `AdaptiveNoise`: Data-dependent noise schedules

#### SDE Types (`src/components/sde/`)
- `VPSDE`: Variance Preserving SDE
- `VESDE`: Variance Exploding SDE
- `SubVPSDE`: Sub-Variance Preserving SDE
- `CustomSDE`: User-defined SDEs

#### Training Objectives (`src/components/objectives/`)
- `ScoreMatching`: Standard score matching
- `Denoising`: Denoising score matching
- `FlowMatching`: Flow-based objectives
- `VariationalBound`: Variational lower bound

#### Samplers (`src/components/samplers/`)
- `EulerSampler`: Euler-Maruyama method
- `HeunSampler`: Heun's method
- `DPMSampler`: DPM-Solver
- `DDIMSampler`: Denoising Diffusion Implicit Models

## Configuration System

### Experiment Configuration Format

```yaml
experiment:
  name: "experiment_name"
  description: "Detailed description"
  
components:
  preprocessing:
    type: "wavelet"
    params:
      wavelet: "db4"
      level: 3
      
  architecture:
    type: "transformer"
    params:
      model_dim: 256
      num_heads: 8
      num_layers: 6
      
  noise:
    type: "standard"
    params:
      schedule: "linear"
      
  sde:
    type: "vpsde"
    params:
      beta_min: 0.1
      beta_max: 20.0
      
  objective:
    type: "score_matching"
    params:
      loss_weight: "uniform"
      
  sampler:
    type: "euler"
    params:
      num_steps: 1000

data:
  dataset: "bitcoin"
  history_len: 252
  predict_len: 21
  
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
```

## Registry Usage

### Registering a New Component

```python
from src.core.registry import Registry
from src.core.base import BasePreprocessor

@Registry.register("preprocessing", "my_preprocessor")
class MyPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def transform(self, data):
        # Implementation
        return processed_data
    
    def inverse_transform(self, data):
        # Implementation
        return original_data
```

### Using Components

```python
from src.core.registry import Registry

# Get a component
preprocessor = Registry.get("preprocessing", "wavelet", wavelet="db4", level=3)

# List available components
available_preprocessors = Registry.list("preprocessing")
```

## Experiment Running

### Command Line Interface

```bash
# Run single experiment
python -m src.run --config configs/experiment.yaml

# Run comparison experiment
python -m src.compare --configs configs/exp1.yaml configs/exp2.yaml

# Run hyperparameter search
python -m src.search --base-config configs/base.yaml --search-config configs/search.yaml
```

### Programmatic Usage

```python
from src.experiments import ExperimentRunner

runner = ExperimentRunner.from_config("configs/experiment.yaml")
results = runner.run()
```

## Data Format

### Input Data Structure
- Time series data should be provided as numpy arrays or torch tensors
- Shape: `(sequence_length, num_features)`
- For univariate series: `(sequence_length, 1)`

### Dataset Requirements
- Must implement `__len__` and `__getitem__`
- Should return dictionaries with keys: `history`, `future`, `metadata`

## Extension Points

### Adding New Components

1. Create a new file in the appropriate component directory
2. Inherit from the corresponding base class
3. Implement required methods
4. Add `@Registry.register` decorator

### Custom Experiments

1. Subclass `BaseExperiment`
2. Override `setup`, `train_step`, `evaluate` methods
3. Register with experiment registry

## Performance Considerations

- Use `torch.compile` for transformer architectures when available
- Enable mixed precision training with `amp`
- Use gradient checkpointing for large models
- Implement efficient batching for variable-length sequences

## Debugging Tips

- Enable debug mode with `--debug` flag
- Use `Registry.inspect(component_type, component_name)` to view component details
- Check component compatibility with `validate_config()`
- Use built-in visualization tools for debugging predictions

## Common Patterns

### Multi-Scale Modeling
```yaml
preprocessing:
  type: "multi_scale"
  params:
    scales: [1, 7, 30]  # Daily, weekly, monthly
```

### Ensemble Methods
```yaml
architecture:
  type: "ensemble"
  params:
    models: ["transformer", "cnn", "rnn"]
    aggregation: "weighted_mean"
```

### Conditional Generation
```yaml
architecture:
  type: "transformer"
  params:
    conditioning: "cross_attention"
    condition_dropout: 0.1  # For classifier-free guidance
```

## Integration with External Tools

- Weights & Biases logging: Set `wandb.enabled: true` in config
- TensorBoard: Logs automatically saved to `experiments/runs/`
- Model serving: Export to ONNX with `export_model()` method