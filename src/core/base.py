"""
Base classes for all pluggable components.

These abstract base classes define the interfaces that all components must implement.
This ensures consistency and interoperability between different implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union, List
import torch
import torch.nn as nn
import numpy as np


class BaseComponent(ABC):
    """Base class for all components."""
    
    def __init__(self, **kwargs):
        """Initialize component with arbitrary keyword arguments."""
        self.config = kwargs
        self._setup()
    
    def _setup(self):
        """Optional setup method called after initialization."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return self.config.copy()


class BasePreprocessor(BaseComponent):
    """Base class for data preprocessing components."""
    
    @abstractmethod
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'BasePreprocessor':
        """
        Fit the preprocessor to training data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform data using fitted parameters.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse transform data back to original space.
        
        Args:
            data: Transformed data
            
        Returns:
            Original-space data
        """
        pass
    
    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output dimension after transformation."""
        pass


class BaseArchitecture(nn.Module, BaseComponent):
    """Base class for model architectures."""
    
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        history_len: int,
        predict_len: int,
        **kwargs
    ):
        nn.Module.__init__(self)
        BaseComponent.__init__(
            self,
            input_dim=input_dim,
            model_dim=model_dim,
            history_len=history_len,
            predict_len=predict_len,
            **kwargs
        )
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.history_len = history_len
        self.predict_len = predict_len
    
    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        x_history: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the architecture.
        
        Args:
            x_t: Noised future values (batch_size, predict_len, input_dim)
            x_history: Historical context (batch_size, history_len, input_dim)
            t: Diffusion timesteps (batch_size,)
            condition: Optional conditioning information
            
        Returns:
            Predicted score/noise (batch_size, predict_len, input_dim)
        """
        pass
    
    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Optional method to compute loss weights based on timestep."""
        return torch.ones_like(t)
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaseNoise(BaseComponent):
    """Base class for noise schedule components."""
    
    @abstractmethod
    def get_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get noise level at timestep t.
        
        Args:
            t: Timesteps in [0, 1]
            
        Returns:
            Noise levels
        """
        pass
    
    @abstractmethod
    def sample_noise(
        self,
        shape: Tuple[int, ...],
        t: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Sample noise for given shape and timestep.
        
        Args:
            shape: Shape of noise to sample
            t: Timesteps
            device: Device to create noise on
            
        Returns:
            Sampled noise
        """
        pass
    
    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative product of (1 - beta) up to timestep t."""
        raise NotImplementedError("Subclass must implement if using discrete schedule")
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise standard deviation at timestep t."""
        raise NotImplementedError("Subclass must implement if using continuous schedule")


class BaseSDE(BaseComponent):
    """Base class for Stochastic Differential Equations."""
    
    @abstractmethod
    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient for forward SDE.
        
        Args:
            x: State
            t: Time
            
        Returns:
            Drift
        """
        pass
    
    @abstractmethod
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """
        Diffusion coefficient for forward SDE.
        
        Args:
            t: Time
            
        Returns:
            Diffusion coefficient
        """
        pass
    
    @abstractmethod
    def transition_kernel(
        self,
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and std of p(x_t | x_0).
        
        Args:
            x0: Initial state
            t: Time
            
        Returns:
            Tuple of (mean, std)
        """
        pass
    
    @abstractmethod
    def prior_sampling(
        self,
        shape: Tuple[int, ...],
        device: torch.device
    ) -> torch.Tensor:
        """
        Sample from prior distribution at t=1.
        
        Args:
            shape: Shape to sample
            device: Device to create samples on
            
        Returns:
            Samples from prior
        """
        pass
    
    def sample_transition(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample from transition kernel p(x_t | x_0)."""
        mean, std = self.transition_kernel(x0, t)
        if noise is None:
            noise = torch.randn_like(x0)
        return mean + std * noise
    
    def reverse_f(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> torch.Tensor:
        """Drift for the reverse SDE."""
        drift = self.f(x, t)
        t_expanded = t.view(-1, *([1] * (x.dim() - 1)))
        diffusion_sq = self.g(t_expanded) ** 2
        return drift - diffusion_sq * score


class BaseObjective(BaseComponent):
    """Base class for training objectives."""
    
    @abstractmethod
    def compute_loss(
        self,
        model: BaseArchitecture,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.
        
        Args:
            model: The model to train
            x_t: Noised data
            x_0: Clean data
            t: Timesteps
            noise: Noise that was added
            condition: Optional conditioning
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        pass
    
    def get_target(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Get the target for the model to predict."""
        # Default: predict noise
        return noise


class BaseSampler(BaseComponent):
    """Base class for sampling algorithms."""
    
    def __init__(self, sde: BaseSDE, **kwargs):
        super().__init__(sde=sde, **kwargs)
        self.sde = sde
    
    @abstractmethod
    def sample(
        self,
        model: BaseArchitecture,
        shape: Tuple[int, ...],
        history: torch.Tensor,
        num_steps: int,
        device: torch.device,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate samples using the model.
        
        Args:
            model: Trained model
            shape: Shape of samples to generate (batch_size, predict_len, input_dim)
            history: Historical context
            num_steps: Number of sampling steps
            device: Device to run on
            condition: Optional conditioning
            guidance_scale: Classifier-free guidance scale
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Generated samples, optionally with trajectory
        """
        pass
    
    def get_update(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Get update for one sampling step."""
        # Default Euler step
        drift = self.sde.reverse_f(x, t, score)
        diffusion = self.sde.g(t)
        
        # Expand dimensions if needed
        if diffusion.dim() < x.dim():
            diffusion = diffusion.view(-1, *([1] * (x.dim() - 1)))
        
        noise = torch.randn_like(x)
        return drift * dt + diffusion * np.sqrt(dt) * noise


class BaseEvaluator(BaseComponent):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def evaluate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictions against targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            metadata: Optional metadata
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def evaluate_ensemble(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble predictions.
        
        Args:
            predictions: Ensemble predictions (n_samples, ...)
            targets: Ground truth
            metadata: Optional metadata
            
        Returns:
            Dictionary of metrics
        """
        # Default: evaluate mean prediction
        mean_pred = predictions.mean(dim=0)
        return self.evaluate(mean_pred, targets, metadata)