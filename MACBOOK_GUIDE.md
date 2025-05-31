# MacBook Setup Guide for Bitcoin Diffusion Model

## 🚀 Quick Start for MacBook

First, navigate to the main project directory:
```bash
cd Diffusion-1
```

### Option 1: MacBook-Optimized Training (Recommended)
```bash
python scripts/train_bitcoin.py --macbook
```

### Option 2: Standard CFG Training
```bash
python scripts/train_bitcoin.py --cfg-experiment
```

### Option 3: Default Training
```bash
python scripts/train_bitcoin.py
```

## 📍 Configuration File Locations

### Main Config Directory:
```
examples/bitcoin_full_implementation/configs/
├── bitcoin_default.yaml         # Standard configuration
├── bitcoin_cfg_experiment.yaml  # CFG research configuration  
└── bitcoin_macbook.yaml         # MacBook MPS-optimized (NEW!)
```

## ⚙️ How to Edit Configurations

### 1. **Quick Edits** (most common):
Navigate to the config file and edit:
```bash
cd Diffusion-1/examples/bitcoin_full_implementation/configs/
nano bitcoin_macbook.yaml        # or use your preferred editor
```

### 2. **Key Settings to Modify**:

**Model Size** (for memory constraints):
```yaml
model:
  model_dim: 192    # Smaller = faster, less memory
  num_heads: 6      # Reduce for speed
  num_layers: 4     # Fewer layers = faster training
  history_len: 126  # Shorter history = less memory
```

**Training Speed**:
```yaml
training:
  batch_size: 16     # Smaller = less memory, slower
  num_epochs: 30     # More epochs = better results
  learning_rate: 0.0002  # Higher = faster convergence
```

**Device Selection**:
```yaml
training:
  device: "auto"     # auto-detects MPS
  # device: "mps"    # Force MPS
  # device: "cpu"    # Force CPU
```

## 🔧 MPS (Metal Performance Shaders) Benefits

- **Faster Training**: ~3-5x speedup vs CPU on M1/M2/M3 MacBooks
- **Native Apple Silicon**: Optimized for Apple's chips
- **Lower Power**: More efficient than CPU training

## 📊 Performance Comparison

| Device | Speed | Memory | Power |
|--------|-------|--------|-------|
| MPS    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| CPU    | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🛠️ Troubleshooting

### If MPS isn't detected:
1. **Check PyTorch version**: Need PyTorch >= 1.12
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **Update PyTorch** (if needed):
   ```bash
   pip install torch>=1.12.0
   ```

3. **Force CPU if MPS issues**:
   Edit config file: `device: "cpu"`

### Memory Issues:
- Reduce `batch_size` (try 8 or 4)
- Reduce `model_dim` (try 128)
- Reduce `history_len` (try 63)

### Slow Training:
- Increase `learning_rate` (try 0.0003)
- Reduce `num_epochs` for testing
- Use `bitcoin_macbook.yaml` config

## 📝 Example Custom Config

Create your own config file:
```bash
cp examples/bitcoin_full_implementation/configs/bitcoin_macbook.yaml \
   examples/bitcoin_full_implementation/configs/my_config.yaml
```

Then edit and use:
```bash
cd Diffusion-1/examples/bitcoin_full_implementation
python scripts/train.py --config configs/my_config.yaml
```

## 🎯 Recommended Settings for Different MacBooks

### M1 MacBook Air (8GB):
```yaml
model:
  model_dim: 128
  num_layers: 3
  history_len: 63
training:
  batch_size: 8
```

### M1/M2 MacBook Pro (16GB+):
```yaml
model:
  model_dim: 192
  num_layers: 4
  history_len: 126
training:
  batch_size: 16
```

### M3 MacBook Pro (32GB+):
```yaml
model:
  model_dim: 256
  num_layers: 6
  history_len: 252
training:
  batch_size: 32
```