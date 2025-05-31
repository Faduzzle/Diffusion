#!/usr/bin/env python3
"""
Wrapper script to prepare Bitcoin data from root directory.

Usage:
    python prepare_data.py
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
    
    # Check if raw data exists
    raw_data = Path("data/raw/bitcoin.csv")
    if not raw_data.exists():
        print(f"Error: Raw data not found at {raw_data}")
        print("Please ensure bitcoin.csv is in bitcoin_diffusion/data/raw/")
        sys.exit(1)
    
    # Run data preparation script
    cmd = [
        sys.executable, "scripts/prepare_data.py",
        "--input", "data/raw/bitcoin.csv",
        "--output", "data/processed/"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    # Execute the script
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\nData preparation completed successfully!")
        print("You can now train the model with:")
        print("  python train_bitcoin.py")
        print("  python train_bitcoin.py --cfg-experiment")
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()