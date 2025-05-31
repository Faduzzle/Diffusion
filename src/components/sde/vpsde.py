"""
Variance Preserving SDE implementations.

Ported from the existing Bitcoin diffusion implementation to the new framework.
"""

import torch
from typing import Tuple

from ...core import BaseSDE, Registry


@Registry.register("sde", "vpsde")
class VPSDE(BaseSDE):
    """
    Variance Preserving SDE (VP-SDE).
    
    The most commonly used SDE for diffusion models, which preserves
    the variance of the data throughout the forward process.
    
    Forward SDE: dx = -0.5 * β(t) * x * dt + √β(t) * dW
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, **kwargs):
        """
        Args:
            beta_min: Minimum beta value at t=0
            beta_max: Maximum beta value at t=1
        """
        super().__init__(beta_min=beta_min, beta_max=beta_max, **kwargs)
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
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        
        alpha_t = self.alpha(t)
        mean = alpha_t * x0
        var = 1 - alpha_t**2
        std = torch.sqrt(var.clamp(min=1e-5))
        
        return mean, std
    
    def prior_sampling(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from standard normal prior at t=1."""
        return torch.randn(shape, device=device)


@Registry.register("sde", "subvpsde")
class SubVPSDE(BaseSDE):
    """
    Sub-Variance Preserving SDE.
    
    A variant of VP-SDE with better numerical properties for discrete sampling.
    Used in improved DDPM formulations.
    """
    
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, **kwargs):
        super().__init__(beta_min=beta_min, beta_max=beta_max, **kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta schedule."""
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative product with different parameterization."""
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