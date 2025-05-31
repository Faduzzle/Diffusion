"""
Standard preprocessing methods for time series.
"""

import torch
import numpy as np
from typing import Union, Optional

from ...core import BasePreprocessor, Registry


@Registry.register("preprocessing", "standard")
class StandardPreprocessor(BasePreprocessor):
    """
    Standard normalization (z-score normalization).
    
    Transforms data to have zero mean and unit variance.
    """
    
    def __init__(self, epsilon: float = 1e-8, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)
        self.epsilon = epsilon
        self.mean_ = None
        self.std_ = None
        self._fitted = False
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'StandardPreprocessor':
        """Compute mean and standard deviation."""
        if isinstance(data, np.ndarray):
            self.mean_ = data.mean(axis=0, keepdims=True)
            self.std_ = data.std(axis=0, keepdims=True) + self.epsilon
        else:
            self.mean_ = data.mean(dim=0, keepdim=True)
            self.std_ = data.std(dim=0, keepdim=True) + self.epsilon
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply standardization."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if isinstance(data, np.ndarray) and isinstance(self.mean_, torch.Tensor):
            # Convert torch stats to numpy
            mean = self.mean_.numpy()
            std = self.std_.numpy()
        elif isinstance(data, torch.Tensor) and isinstance(self.mean_, np.ndarray):
            # Convert numpy stats to torch
            mean = torch.from_numpy(self.mean_).to(data.device)
            std = torch.from_numpy(self.std_).to(data.device)
        else:
            mean = self.mean_
            std = self.std_
        
        return (data - mean) / std
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse standardization."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        if isinstance(data, np.ndarray) and isinstance(self.mean_, torch.Tensor):
            mean = self.mean_.numpy()
            std = self.std_.numpy()
        elif isinstance(data, torch.Tensor) and isinstance(self.mean_, np.ndarray):
            mean = torch.from_numpy(self.mean_).to(data.device)
            std = torch.from_numpy(self.std_).to(data.device)
        else:
            mean = self.mean_
            std = self.std_
        
        return data * std + mean
    
    @property
    def output_dim(self) -> int:
        """Output dimension is same as input."""
        if self.mean_ is None:
            return 1
        return self.mean_.shape[-1] if len(self.mean_.shape) > 1 else 1


@Registry.register("preprocessing", "minmax")
class MinMaxPreprocessor(BasePreprocessor):
    """
    Min-Max normalization.
    
    Scales data to [0, 1] range.
    """
    
    def __init__(self, feature_range: tuple = (0, 1), epsilon: float = 1e-8, **kwargs):
        super().__init__(feature_range=feature_range, epsilon=epsilon, **kwargs)
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_ = None
        self.max_ = None
        self._fitted = False
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'MinMaxPreprocessor':
        """Compute min and max values."""
        if isinstance(data, np.ndarray):
            self.min_ = data.min(axis=0, keepdims=True)
            self.max_ = data.max(axis=0, keepdims=True)
        else:
            self.min_ = data.min(dim=0, keepdim=True)[0]
            self.max_ = data.max(dim=0, keepdim=True)[0]
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply min-max scaling."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Handle type conversions
        if isinstance(data, np.ndarray) and isinstance(self.min_, torch.Tensor):
            min_val = self.min_.numpy()
            max_val = self.max_.numpy()
        elif isinstance(data, torch.Tensor) and isinstance(self.min_, np.ndarray):
            min_val = torch.from_numpy(self.min_).to(data.device)
            max_val = torch.from_numpy(self.max_).to(data.device)
        else:
            min_val = self.min_
            max_val = self.max_
        
        # Scale to [0, 1]
        scaled = (data - min_val) / (max_val - min_val + self.epsilon)
        
        # Scale to feature_range
        if self.feature_range != (0, 1):
            scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return scaled
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Inverse min-max scaling."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        # Handle type conversions
        if isinstance(data, np.ndarray) and isinstance(self.min_, torch.Tensor):
            min_val = self.min_.numpy()
            max_val = self.max_.numpy()
        elif isinstance(data, torch.Tensor) and isinstance(self.min_, np.ndarray):
            min_val = torch.from_numpy(self.min_).to(data.device)
            max_val = torch.from_numpy(self.max_).to(data.device)
        else:
            min_val = self.min_
            max_val = self.max_
        
        # Unscale from feature_range
        if self.feature_range != (0, 1):
            data = (data - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        
        # Unscale from [0, 1]
        return data * (max_val - min_val + self.epsilon) + min_val
    
    @property
    def output_dim(self) -> int:
        """Output dimension is same as input."""
        if self.min_ is None:
            return 1
        return self.min_.shape[-1] if len(self.min_.shape) > 1 else 1


@Registry.register("preprocessing", "log_returns")
class LogReturnsPreprocessor(BasePreprocessor):
    """
    Log returns preprocessing for financial time series.
    
    Converts prices to log returns and applies normalization.
    """
    
    def __init__(self, normalize: bool = True, epsilon: float = 1e-8, **kwargs):
        super().__init__(normalize=normalize, epsilon=epsilon, **kwargs)
        self.normalize = normalize
        self.epsilon = epsilon
        self.normalizer = StandardPreprocessor(epsilon=epsilon) if normalize else None
        self._last_price = None
        self._fitted = False
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'LogReturnsPreprocessor':
        """Compute log returns and fit normalizer."""
        # Store last price for inverse transform
        self._last_price = data[-1:] if isinstance(data, np.ndarray) else data[-1:].clone()
        
        # Compute log returns
        if isinstance(data, np.ndarray):
            log_returns = np.log(data[1:] / data[:-1] + self.epsilon)
        else:
            log_returns = torch.log(data[1:] / data[:-1] + self.epsilon)
        
        # Fit normalizer on log returns
        if self.normalizer:
            self.normalizer.fit(log_returns)
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert to log returns and normalize."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Compute log returns
        if isinstance(data, np.ndarray):
            log_returns = np.log(data[1:] / data[:-1] + self.epsilon)
        else:
            log_returns = torch.log(data[1:] / data[:-1] + self.epsilon)
        
        # Apply normalization
        if self.normalizer:
            log_returns = self.normalizer.transform(log_returns)
        
        return log_returns
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Convert log returns back to prices."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        # Denormalize if needed
        if self.normalizer:
            data = self.normalizer.inverse_transform(data)
        
        # Convert log returns to prices
        if isinstance(data, np.ndarray):
            prices = np.exp(np.cumsum(data, axis=0))
            # Scale by last known price
            if isinstance(self._last_price, torch.Tensor):
                last_price = self._last_price.numpy()
            else:
                last_price = self._last_price
            prices = prices * last_price
        else:
            prices = torch.exp(torch.cumsum(data, dim=0))
            # Scale by last known price
            if isinstance(self._last_price, np.ndarray):
                last_price = torch.from_numpy(self._last_price).to(data.device)
            else:
                last_price = self._last_price.to(data.device)
            prices = prices * last_price
        
        return prices
    
    @property
    def output_dim(self) -> int:
        """Output dimension is same as input."""
        return 1  # Log returns are univariate for each series