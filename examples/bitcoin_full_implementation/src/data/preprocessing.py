"""
Data preprocessing utilities for Bitcoin time series.

Handles data loading, cleaning, normalization, and feature engineering
for financial time series data.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config
        self.normalization_params = {}
        self.metadata = {
            'processed_date': datetime.now().isoformat(),
            'config': config
        }
    
    def load_csv(self, filepath: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load CSV file with proper parsing.
        
        Args:
            filepath: Path to CSV file
            columns: Specific columns to load
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        # Try to infer date column
        df = pd.read_csv(filepath, nrows=5)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_columns:
            df = pd.read_csv(filepath, parse_dates=date_columns, index_col=date_columns[0])
        else:
            df = pd.read_csv(filepath)
        
        # Select specific columns if requested
        if columns:
            df = df[columns]
        
        # Handle missing values
        fill_method = self.config.get('fill_method', 'ffill')
        if fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'interpolate':
            df = df.interpolate()
        
        return df
    
    def compute_returns(self, prices: Union[pd.Series, np.ndarray, torch.Tensor], 
                       method: str = 'log') -> Union[pd.Series, np.ndarray, torch.Tensor]:
        """
        Compute returns from price series.
        
        Args:
            prices: Price series
            method: 'log' for log returns, 'simple' for arithmetic returns
            
        Returns:
            Returns series (first value is 0)
        """
        if isinstance(prices, pd.Series):
            if method == 'log':
                returns = np.log(prices / prices.shift(1))
            else:
                returns = prices.pct_change()
            returns.iloc[0] = 0
            return returns
        
        elif isinstance(prices, np.ndarray):
            if method == 'log':
                returns = np.log(prices[1:] / prices[:-1])
            else:
                returns = (prices[1:] - prices[:-1]) / prices[:-1]
            returns = np.concatenate([[0], returns])
            return returns
        
        elif isinstance(prices, torch.Tensor):
            if method == 'log':
                returns = torch.log(prices[1:] / prices[:-1])
            else:
                returns = (prices[1:] - prices[:-1]) / prices[:-1]
            returns = torch.cat([torch.zeros(1, device=prices.device), returns])
            return returns
    
    def normalize(self, data: torch.Tensor, method: str = 'standard', 
                  fit: bool = True) -> torch.Tensor:
        """
        Normalize data using specified method.
        
        Args:
            data: Data to normalize
            method: Normalization method
            fit: Whether to fit normalization parameters
            
        Returns:
            Normalized data
        """
        if fit:
            if method == 'standard':
                mean = data.mean(dim=0, keepdim=True)
                std = data.std(dim=0, keepdim=True)
                std = torch.where(std == 0, torch.ones_like(std), std)
                self.normalization_params = {'mean': mean, 'std': std}
                
            elif method == 'minmax':
                min_val = data.min(dim=0, keepdim=True)[0]
                max_val = data.max(dim=0, keepdim=True)[0]
                range_val = max_val - min_val
                range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
                self.normalization_params = {'min': min_val, 'range': range_val}
                
            elif method == 'robust':
                median = data.median(dim=0, keepdim=True)[0]
                mad = (data - median).abs().median(dim=0, keepdim=True)[0]
                mad = torch.where(mad == 0, torch.ones_like(mad), mad)
                self.normalization_params = {'median': median, 'mad': mad}
                
            elif method == 'max':
                max_abs = data.abs().max(dim=0, keepdim=True)[0]
                max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
                self.normalization_params = {'max_abs': max_abs}
            
            self.metadata['normalization'] = {
                'method': method,
                'params': {k: v.tolist() for k, v in self.normalization_params.items()}
            }
        
        # Apply normalization
        if method == 'standard':
            return (data - self.normalization_params['mean']) / self.normalization_params['std']
        elif method == 'minmax':
            return (data - self.normalization_params['min']) / self.normalization_params['range']
        elif method == 'robust':
            return (data - self.normalization_params['median']) / self.normalization_params['mad']
        elif method == 'max':
            return data / self.normalization_params['max_abs']
        elif method == 'none':
            return data
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denormalize(self, data: torch.Tensor, method: Optional[str] = None) -> torch.Tensor:
        """
        Inverse normalization transformation.
        
        Args:
            data: Normalized data
            method: Normalization method (uses stored if not provided)
            
        Returns:
            Denormalized data
        """
        if method is None:
            method = self.metadata.get('normalization', {}).get('method', 'none')
        
        if method == 'standard':
            return data * self.normalization_params['std'] + self.normalization_params['mean']
        elif method == 'minmax':
            return data * self.normalization_params['range'] + self.normalization_params['min']
        elif method == 'robust':
            return data * self.normalization_params['mad'] + self.normalization_params['median']
        elif method == 'max':
            return data * self.normalization_params['max_abs']
        elif method == 'none':
            return data
    
    def compute_volatility(self, returns: torch.Tensor, window: int = 21) -> torch.Tensor:
        """
        Compute rolling volatility.
        
        Args:
            returns: Return series
            window: Rolling window size
            
        Returns:
            Volatility series
        """
        # Pad returns for rolling window
        padded = torch.cat([torch.zeros(window - 1), returns])
        
        # Compute rolling std
        volatility = torch.zeros_like(returns)
        for i in range(len(returns)):
            window_returns = padded[i:i + window]
            volatility[i] = window_returns.std()
        
        return volatility
    
    def create_features(self, prices: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Create feature set from price data.
        
        Args:
            prices: DataFrame with price columns
            
        Returns:
            Dictionary of feature tensors
        """
        features = {}
        
        # Convert to tensor
        price_tensor = torch.tensor(prices.values, dtype=torch.float32)
        
        # Log returns
        returns = self.compute_returns(price_tensor, method='log')
        features['returns'] = returns
        
        # Volatility
        if self.config.get('include_volatility', True):
            vol_window = self.config.get('volatility_window', 21)
            features['volatility'] = self.compute_volatility(returns, window=vol_window)
        
        # Technical indicators
        if self.config.get('include_technical', False):
            # Simple moving averages
            for window in [5, 20, 50]:
                sma = self._compute_sma(price_tensor, window)
                features[f'sma_{window}'] = sma
            
            # RSI
            features['rsi'] = self._compute_rsi(price_tensor)
        
        return features
    
    def _compute_sma(self, prices: torch.Tensor, window: int) -> torch.Tensor:
        """Compute simple moving average."""
        padded = torch.cat([torch.full((window - 1,), prices[0]), prices])
        sma = torch.zeros_like(prices)
        
        for i in range(len(prices)):
            sma[i] = padded[i:i + window].mean()
        
        return sma / prices - 1  # Return as ratio to current price
    
    def _compute_rsi(self, prices: torch.Tensor, window: int = 14) -> torch.Tensor:
        """Compute Relative Strength Index."""
        deltas = prices[1:] - prices[:-1]
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        # Pad for first value
        gains = torch.cat([torch.zeros(1), gains])
        losses = torch.cat([torch.zeros(1), losses])
        
        # Exponential moving average
        avg_gains = torch.zeros_like(prices)
        avg_losses = torch.zeros_like(prices)
        
        avg_gains[window] = gains[:window].mean()
        avg_losses[window] = losses[:window].mean()
        
        for i in range(window + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (window - 1) + gains[i]) / window
            avg_losses[i] = (avg_losses[i-1] * (window - 1) + losses[i]) / window
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi / 100  # Normalize to [0, 1]
    
    def save_processed_data(self, data: torch.Tensor, filepath: Union[str, Path]):
        """
        Save processed data with metadata.
        
        Args:
            data: Processed data tensor
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PyTorch file with metadata
        torch.save({
            'data': data,
            'metadata': self.metadata,
            'normalization_params': self.normalization_params
        }, filepath)
        
        # Also save metadata as JSON for easy inspection
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_processed_data(self, filepath: Union[str, Path]) -> Tuple[torch.Tensor, Dict]:
        """
        Load processed data with metadata.
        
        Args:
            filepath: Path to processed data file
            
        Returns:
            Tuple of (data tensor, metadata dict)
        """
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, weights_only=False)
        
        self.normalization_params = checkpoint.get('normalization_params', {})
        self.metadata = checkpoint.get('metadata', {})
        
        return checkpoint['data'], self.metadata


def prepare_bitcoin_data(
    raw_data_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Dict[str, Path]:
    """
    Complete pipeline to prepare Bitcoin data for training.
    
    Args:
        raw_data_path: Path to raw Bitcoin CSV
        output_dir: Directory for processed outputs
        config: Preprocessing configuration
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        
    Returns:
        Dictionary with paths to train/val/test data files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load raw data
    df = preprocessor.load_csv(raw_data_path)
    
    # Use closing price (adjust column name as needed)
    price_col = 'Close' if 'Close' in df.columns else df.columns[0]
    prices = df[price_col]
    
    # Compute returns
    returns = preprocessor.compute_returns(prices, method='log')
    
    # Convert to tensor
    returns_tensor = torch.tensor(returns.values, dtype=torch.float32).unsqueeze(-1)
    
    # Split data
    n = len(returns_tensor)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = returns_tensor[:train_end]
    val_data = returns_tensor[train_end:val_end]
    test_data = returns_tensor[val_end:]
    
    # Normalize using training data statistics
    train_normalized = preprocessor.normalize(train_data, method=config.get('norm_method', 'standard'))
    val_normalized = preprocessor.normalize(val_data, fit=False)
    test_normalized = preprocessor.normalize(test_data, fit=False)
    
    # Save processed data
    output_paths = {}
    for split, data in [('train', train_normalized), ('val', val_normalized), ('test', test_normalized)]:
        filepath = output_dir / f'bitcoin_{split}.pt'
        preprocessor.save_processed_data(data, filepath)
        output_paths[split] = filepath
    
    return output_paths