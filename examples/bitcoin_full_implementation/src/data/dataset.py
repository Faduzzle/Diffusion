"""
Dataset classes for Bitcoin time series data.

Implements sliding window datasets with various preprocessing options
for training diffusion models on financial time series.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Dict, Any
import pandas as pd


class SlidingWindowDataset(Dataset):
    """
    Creates sliding windows over time series data for training.
    
    Each sample consists of:
    - Historical window: Used as conditioning information
    - Future window: Target for prediction (will be noised during training)
    """
    
    def __init__(
        self,
        data: torch.Tensor,
        history_len: int = 252,  # ~1 year of trading days
        predict_len: int = 21,   # ~1 month ahead
        stride: int = 1,         # Step size between windows
        mask_prob: float = 0.0,  # Probability of masking historical data
        max_samples: Optional[int] = None,  # Limit dataset size
    ):
        """
        Args:
            data: Time series data of shape (seq_len, channels)
            history_len: Length of historical window
            predict_len: Length of prediction window
            stride: Step size between consecutive windows
            mask_prob: Probability of masking historical values (for robustness)
            max_samples: Maximum number of samples to generate
        """
        super().__init__()
        
        self.data = data
        self.history_len = history_len
        self.predict_len = predict_len
        self.stride = stride
        self.mask_prob = mask_prob
        
        # Calculate valid indices for sliding windows
        self.total_len = history_len + predict_len
        self.valid_starts = list(range(0, len(data) - self.total_len + 1, stride))
        
        # Limit dataset size if specified
        if max_samples is not None and len(self.valid_starts) > max_samples:
            # Randomly sample indices
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(self.valid_starts), max_samples, replace=False)
            self.valid_starts = [self.valid_starts[i] for i in sorted(indices)]
    
    def __len__(self) -> int:
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - 'history': Historical window (history_len, channels)
            - 'future': Future window (predict_len, channels)
            - 'start_idx': Starting index in original data
        """
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.total_len
        
        # Extract windows
        history = self.data[start_idx:start_idx + self.history_len].clone()
        future = self.data[start_idx + self.history_len:end_idx].clone()
        
        # Optional: Apply masking to history for robustness
        if self.mask_prob > 0 and np.random.rand() < self.mask_prob:
            mask = torch.rand(self.history_len) > self.mask_prob
            history = history * mask.unsqueeze(-1)
        
        return {
            'history': history,
            'future': future,
            'start_idx': start_idx
        }


class FinancialDataset(SlidingWindowDataset):
    """
    Extended dataset class with financial-specific features.
    
    Adds support for:
    - Multiple price series alignment
    - Volume data
    - Technical indicators
    - Market regime labels
    """
    
    def __init__(
        self,
        price_data: torch.Tensor,
        volume_data: Optional[torch.Tensor] = None,
        indicators: Optional[Dict[str, torch.Tensor]] = None,
        regime_labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Args:
            price_data: Price/return data
            volume_data: Optional volume data
            indicators: Dictionary of technical indicators
            regime_labels: Optional market regime classifications
            **kwargs: Arguments passed to SlidingWindowDataset
        """
        # Combine all features
        features = [price_data]
        self.feature_names = ['price']
        self.feature_dims = {'price': price_data.shape[-1] if price_data.dim() > 1 else 1}
        
        if volume_data is not None:
            features.append(volume_data)
            self.feature_names.append('volume')
            self.feature_dims['volume'] = volume_data.shape[-1] if volume_data.dim() > 1 else 1
        
        if indicators:
            for name, data in indicators.items():
                features.append(data)
                self.feature_names.append(name)
                self.feature_dims[name] = data.shape[-1] if data.dim() > 1 else 1
        
        # Stack all features
        if len(features) > 1:
            # Ensure all tensors have same length
            min_len = min(f.shape[0] for f in features)
            features = [f[:min_len] for f in features]
            
            # Concatenate along feature dimension
            data = torch.cat([f.unsqueeze(-1) if f.dim() == 1 else f for f in features], dim=-1)
        else:
            data = features[0]
            if data.dim() == 1:
                data = data.unsqueeze(-1)
        
        self.regime_labels = regime_labels
        
        super().__init__(data, **kwargs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with additional financial metadata."""
        sample = super().__getitem__(idx)
        
        # Add regime labels if available
        if self.regime_labels is not None:
            start_idx = sample['start_idx']
            sample['regime'] = self.regime_labels[start_idx:start_idx + self.total_len]
        
        # Add feature information
        sample['feature_dims'] = self.feature_dims
        sample['feature_names'] = self.feature_names
        
        return sample


class MultiAssetDataset(Dataset):
    """
    Dataset for multiple correlated assets.
    
    Handles alignment and synchronization of multiple time series,
    useful for modeling cross-asset dependencies.
    """
    
    def __init__(
        self,
        assets_data: Dict[str, torch.Tensor],
        history_len: int = 252,
        predict_len: int = 21,
        stride: int = 1,
        correlation_threshold: float = 0.3,
    ):
        """
        Args:
            assets_data: Dictionary mapping asset names to time series
            history_len: Length of historical window
            predict_len: Length of prediction window  
            stride: Step size between windows
            correlation_threshold: Min correlation to include asset pairs
        """
        super().__init__()
        
        self.asset_names = list(assets_data.keys())
        self.history_len = history_len
        self.predict_len = predict_len
        self.stride = stride
        
        # Align all series to same length
        min_len = min(data.shape[0] for data in assets_data.values())
        self.data = {}
        for name, data in assets_data.items():
            self.data[name] = data[:min_len]
            if self.data[name].dim() == 1:
                self.data[name] = self.data[name].unsqueeze(-1)
        
        # Calculate correlations
        self.correlations = self._calculate_correlations(correlation_threshold)
        
        # Valid window starts
        self.total_len = history_len + predict_len
        self.valid_starts = list(range(0, min_len - self.total_len + 1, stride))
    
    def _calculate_correlations(self, threshold: float) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise correlations above threshold."""
        correlations = {}
        
        for i, asset1 in enumerate(self.asset_names):
            for asset2 in self.asset_names[i+1:]:
                data1 = self.data[asset1].squeeze()
                data2 = self.data[asset2].squeeze()
                
                # Calculate correlation
                corr = torch.corrcoef(torch.stack([data1, data2]))[0, 1].item()
                
                if abs(corr) > threshold:
                    correlations[(asset1, asset2)] = corr
        
        return correlations
    
    def __len__(self) -> int:
        return len(self.valid_starts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get multi-asset sample.
        
        Returns:
            Dictionary with data for each asset and correlation info
        """
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.total_len
        
        sample = {
            'assets': {},
            'correlations': self.correlations,
            'start_idx': start_idx
        }
        
        for asset_name in self.asset_names:
            history = self.data[asset_name][start_idx:start_idx + self.history_len]
            future = self.data[asset_name][start_idx + self.history_len:end_idx]
            
            sample['assets'][asset_name] = {
                'history': history,
                'future': future
            }
        
        return sample