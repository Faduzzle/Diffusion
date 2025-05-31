#!/usr/bin/env python3
"""
Wrapper script to train Bitcoin diffusion model from root directory.

Usage:
    python train_bitcoin.py [--cfg-experiment]
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Change to bitcoin_diffusion directory
    bitcoin_dir = Path(__file__).parent / "bitcoin_diffusion"
    
    if not bitcoin_dir.exists():
        print("Error: bitcoin_diffusion directory not found!")
        sys.exit(1)
    
    # Change to bitcoin_diffusion directory
    os.chdir(bitcoin_dir)
    
    # Determine config file
    if len(sys.argv) > 1:
        if sys.argv[1] == "--cfg-experiment":
            config_file = "configs/bitcoin_cfg_experiment.yaml"
            print("Using CFG experiment configuration...")
        elif sys.argv[1] == "--macbook":
            config_file = "configs/bitcoin_macbook.yaml"
            print("Using MacBook-optimized configuration with MPS support...")
        else:
            config_file = "configs/bitcoin_default.yaml"
            print("Using default configuration...")
    else:
        config_file = "configs/bitcoin_default.yaml"
        print("Using default configuration...")
    
    # Run training script
    cmd = [sys.executable, "scripts/train.py", "--config", config_file]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    # Execute the training script
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()