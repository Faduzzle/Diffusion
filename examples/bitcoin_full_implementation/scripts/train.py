#!/usr/bin/env python3
"""
Train Bitcoin diffusion model.

Usage:
    python train.py --config configs/bitcoin_default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, ConfigValidator
from src.training import create_trainer


def main():
    parser = argparse.ArgumentParser(description='Train Bitcoin diffusion model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate configuration
    ConfigValidator.validate_full_config(config)
    
    # Set device intelligently
    import torch
    current_device = config['training'].get('device', 'auto')
    
    if current_device == 'auto':
        if args.gpu >= 0 and torch.cuda.is_available():
            config['training']['device'] = f'cuda:{args.gpu}'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config['training']['device'] = 'mps'
            print("Using MPS (Metal Performance Shaders) for MacBook acceleration")
        else:
            config['training']['device'] = 'cpu'
            print("Using CPU (no GPU acceleration available)")
    elif current_device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to MPS or CPU")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config['training']['device'] = 'mps'
        else:
            config['training']['device'] = 'cpu'
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    num_epochs = config['training']['num_epochs']
    history = trainer.train(num_epochs)
    
    print("Training completed!")
    print(f"Final train loss: {history['train']['loss'][-1]:.4f}")
    if 'val' in history and 'loss' in history['val']:
        print(f"Final val loss: {history['val']['loss'][-1]:.4f}")


if __name__ == '__main__':
    main()