"""
Stochastic Differential Equations for Diffusion Models.

Implements various SDEs that define the forward (noising) and reverse (denoising)
processes in continuous-time diffusion models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class SDE(ABC):
    """
    Abstract base class for Stochastic Differential Equations.
    
    Defines the interface for SDEs used in diffusion models:
    dx = f(x,t)dt + g(t)dW
    
    Where:
    - f(x,t): drift coefficient
    - g(t): diffusion coefficient
    - W: Brownian motion
    """
    
    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Noise schedule function β(t)."""
        pass
    
    @abstractmethod
    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient for forward SDE."""
        pass
    
    @abstractmethod
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient for forward SDE."""
        pass
    
    @abstractmethod
    def transition_kernel(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get parameters of p(x_t | x_0).
        
        Returns:
            mean: Mean of the transition kernel
            std: Standard deviation of the transition kernel
        """
        pass
    
    def sample_transition(self, x0: torch.Tensor, t: torch.Tensor, 
                         noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the transition kernel p(x_t | x_0).
        
        Args:
            x0: Initial state
            t: Time step
            noise: Optional pre-sampled noise
            
        Returns:
            Sampled state at time t
        """
        mean, std = self.transition_kernel(x0, t)
        if noise is None:
            noise = torch.randn_like(x0)
        return mean + std * noise
    
    @abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from the prior distribution at t=1."""
        pass


class VPSDE(SDE):
    """
    Variance Preserving SDE (VP-SDE).
    
    The most commonly used SDE for diffusion models, which preserves
    the variance of the data throughout the forward process.
    
    Forward SDE: dx = -0.5 * β(t) * x * dt + √β(t) * dW
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        """
        Args:
            beta_min: Minimum beta value at t=0
            beta_max: Maximum beta value at t=1
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative product of (1 - beta)."""
        log_alpha = -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)
        return torch.exp(log_alpha)
    
    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient: f(x,t) = -0.5 * β(t) * x."""
        # Expand t to match x dimensions
        t = t.view(-1, *([1] * (x.dim() - 1)))
        return -0.5 * self.beta(t) * x
    
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient: g(t) = √β(t)."""
        return torch.sqrt(self.beta(t))
    
    def transition_kernel(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get mean and std of p(x_t | x_0).
        
        For VP-SDE:
        - mean = α(t) * x0
        - std = √(1 - α(t)²)
        """
        # Expand t to match x0 dimensions
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        
        alpha_t = self.alpha(t)
        mean = alpha_t * x0
        var = 1 - alpha_t**2
        std = torch.sqrt(var.clamp(min=1e-5))  # Numerical stability
        
        return mean, std
    
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from standard normal prior at t=1."""
        return torch.randn(shape, device=device)
    
    def reverse_f(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """
        Drift for the reverse SDE.
        
        Reverse drift: f(x,t) - g(t)² * score(x,t)
        """
        drift = self.f(x, t)
        t_expanded = t.view(-1, *([1] * (x.dim() - 1)))
        diffusion_sq = self.g(t_expanded)**2
        return drift - diffusion_sq * score


class VESDE(SDE):
    """
    Variance Exploding SDE (VE-SDE).
    
    An alternative formulation where variance grows without bound.
    Often used for image generation tasks.
    
    Forward SDE: dx = √(dσ²/dt) * dW
    """
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Exponential sigma schedule."""
        log_sigma = torch.log(torch.tensor(self.sigma_min)) + \
                   t * (torch.log(torch.tensor(self.sigma_max)) - torch.log(torch.tensor(self.sigma_min)))
        return torch.exp(log_sigma)
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Not used in VE-SDE, but included for interface compatibility."""
        # Approximate beta for VE-SDE
        sigma_t = self.sigma(t)
        return sigma_t * torch.sqrt(2 * torch.log(torch.tensor(self.sigma_max / self.sigma_min)))
    
    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """VE-SDE has zero drift."""
        return torch.zeros_like(x)
    
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient for VE-SDE."""
        # g(t) = σ(t) * √(2log(σ_max/σ_min))
        return self.sigma(t) * torch.sqrt(2 * torch.log(torch.tensor(self.sigma_max / self.sigma_min)))
    
    def transition_kernel(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For VE-SDE:
        - mean = x0 (no scaling)
        - std = σ(t)
        """
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        sigma_t = self.sigma(t)
        return x0, sigma_t
    
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from N(0, σ_max²) prior."""
        return torch.randn(shape, device=device) * self.sigma_max


class SubVPSDE(SDE):
    """
    Sub-Variance Preserving SDE.
    
    A variant of VP-SDE with better numerical properties for discrete sampling.
    Used in improved DDPM formulations.
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative product with different parameterization."""
        # Use quadratic schedule for smoother transitions
        log_alpha = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return torch.exp(log_alpha)
    
    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Modified drift for sub-VP."""
        t = t.view(-1, *([1] * (x.dim() - 1)))
        beta_t = self.beta(t)
        return -0.5 * beta_t * x / (1 - self.alpha(t)**2 + 1e-5)
    
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Modified diffusion coefficient."""
        beta_t = self.beta(t)
        alpha_t = self.alpha(t)
        return torch.sqrt(beta_t * (1 - alpha_t**2) / (1 - alpha_t**2 + beta_t))
    
    def transition_kernel(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Similar to VP-SDE but with modified variance."""
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        alpha_t = self.alpha(t)
        mean = alpha_t * x0
        var = 1 - alpha_t**2
        std = torch.sqrt(var.clamp(min=1e-5))
        return mean, std
    
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Standard normal prior."""
        return torch.randn(shape, device=device)


def create_sde(sde_type: str = 'vpsde', **kwargs) -> SDE:
    """
    Factory function to create SDE instances.
    
    Args:
        sde_type: Type of SDE ('vpsde', 'vesde', 'subvpsde')
        **kwargs: Arguments passed to SDE constructor
        
    Returns:
        SDE instance
    """
    sde_types = {
        'vpsde': VPSDE,
        'vesde': VESDE,
        'subvpsde': SubVPSDE
    }
    
    if sde_type not in sde_types:
        raise ValueError(f"Unknown SDE type: {sde_type}. Available: {list(sde_types.keys())}")
    
    return sde_types[sde_type](**kwargs)