"""
Variance Exploding SDE implementation.
"""

import torch
from typing import Tuple

from ...core import BaseSDE, Registry


@Registry.register("sde", "vesde")
class VESDE(BaseSDE):
    """
    Variance Exploding SDE (VE-SDE).
    
    An alternative formulation where variance grows without bound.
    Often used for image generation tasks.
    
    Forward SDE: dx = √(dσ²/dt) * dW
    """
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, **kwargs):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        super().__init__(sigma_min=sigma_min, sigma_max=sigma_max, **kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Exponential sigma schedule."""
        log_sigma = torch.log(torch.tensor(self.sigma_min)) + \
                   t * (torch.log(torch.tensor(self.sigma_max)) - torch.log(torch.tensor(self.sigma_min)))
        return torch.exp(log_sigma)
    
    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """VE-SDE has zero drift."""
        return torch.zeros_like(x)
    
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient for VE-SDE."""
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