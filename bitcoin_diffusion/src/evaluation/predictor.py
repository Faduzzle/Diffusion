"""
Prediction generation for trained diffusion models.

Implements various sampling algorithms for generating predictions
from the reverse diffusion process.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm

from ..models import DiffusionTransformer
from ..models.sde import SDE, create_sde
from ..data import DataPreprocessor


class DiffusionPredictor:
    """Generate predictions from trained diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        sde: SDE,
        preprocessor: Optional[DataPreprocessor] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: Trained diffusion model
            sde: SDE defining the diffusion process
            preprocessor: Data preprocessor for denormalization
            device: Computation device
        """
        self.model = model
        self.sde = sde
        self.preprocessor = preprocessor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], device: Optional[torch.device] = None):
        """
        Load predictor from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Computation device
            
        Returns:
            Configured predictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create model
        from ..models import create_model
        model = create_model(config['model'])
        
        # Load weights (use EMA if available)
        if 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create SDE
        sde = create_sde(**config['sde'])
        
        # Create preprocessor
        preprocessor = None
        if 'normalization_params' in checkpoint:
            preprocessor = DataPreprocessor(config.get('data', {}))
            preprocessor.normalization_params = checkpoint['normalization_params']
            preprocessor.metadata = checkpoint.get('metadata', {})
        
        return cls(model, sde, preprocessor, device)
    
    def euler_maruyama_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        history: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Single step of Euler-Maruyama solver.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step size
            history: Historical context
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Next state
        """
        batch_size = x.shape[0]
        
        # Get score prediction
        with torch.no_grad():
            score = self.model(x, history, t)
            
            # Classifier-free guidance
            if guidance_scale != 1.0 and hasattr(self.model, 'get_uncond_score'):
                uncond_score = self.model.get_uncond_score(x, history, t)
                score = uncond_score + guidance_scale * (score - uncond_score)
        
        # Reverse drift
        drift = self.sde.f(x, t)
        t_expanded = t.view(-1, *([1] * (x.dim() - 1)))
        diffusion = self.sde.g(t_expanded)
        reverse_drift = drift - diffusion**2 * score
        
        # Add noise
        noise = torch.randn_like(x) * np.sqrt(dt)
        
        # Update
        x_next = x - reverse_drift * dt + diffusion * noise
        
        return x_next
    
    def heun_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        history: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Heun's method (improved Euler) for better accuracy.
        
        Args:
            x: Current state
            t: Current time
            dt: Time step size
            history: Historical context
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Next state
        """
        # First stage (Euler step)
        x_euler = self.euler_maruyama_step(x, t, dt, history, guidance_scale)
        
        # Second stage (correction)
        t_next = t - dt
        with torch.no_grad():
            score_next = self.model(x_euler, history, t_next)
            
            if guidance_scale != 1.0 and hasattr(self.model, 'get_uncond_score'):
                uncond_score_next = self.model.get_uncond_score(x_euler, history, t_next)
                score_next = uncond_score_next + guidance_scale * (score_next - uncond_score_next)
        
        # Average the drifts
        score = self.model(x, history, t)
        if guidance_scale != 1.0 and hasattr(self.model, 'get_uncond_score'):
            uncond_score = self.model.get_uncond_score(x, history, t)
            score = uncond_score + guidance_scale * (score - uncond_score)
        
        # Compute drifts
        t_expanded = t.view(-1, *([1] * (x.dim() - 1)))
        t_next_expanded = t_next.view(-1, *([1] * (x.dim() - 1)))
        
        drift1 = self.sde.f(x, t) - self.sde.g(t_expanded)**2 * score
        drift2 = self.sde.f(x_euler, t_next) - self.sde.g(t_next_expanded)**2 * score_next
        
        # Heun update
        noise = torch.randn_like(x) * np.sqrt(dt)
        x_next = x - 0.5 * (drift1 + drift2) * dt + self.sde.g(t_expanded) * noise
        
        return x_next
    
    def sample_trajectories(
        self,
        history: torch.Tensor,
        num_samples: int = 100,
        num_steps: int = 1000,
        solver: str = 'euler',
        guidance_scale: float = 1.0,
        return_all_steps: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate future trajectories using reverse diffusion.
        
        Args:
            history: Historical data (batch_size, history_len, features)
            num_samples: Number of trajectories per history
            num_steps: Number of diffusion steps
            solver: Solver type ('euler', 'heun')
            guidance_scale: Classifier-free guidance scale
            return_all_steps: Whether to return intermediate steps
            
        Returns:
            Generated trajectories (batch_size, num_samples, predict_len, features)
            or list of all intermediate steps if return_all_steps=True
        """
        batch_size = history.shape[0]
        predict_len = self.model.predict_len
        features = history.shape[-1]
        
        # Expand history for multiple samples
        history_expanded = history.unsqueeze(1).expand(-1, num_samples, -1, -1)
        history_flat = history_expanded.reshape(batch_size * num_samples, *history.shape[1:])
        
        # Initialize from prior
        x = self.sde.prior_sampling(
            (batch_size * num_samples, predict_len, features),
            device=self.device
        )
        
        # Time steps
        dt = 1.0 / num_steps
        timesteps = torch.linspace(1, 0, num_steps + 1, device=self.device)
        
        # Storage for all steps
        if return_all_steps:
            all_steps = [x.clone()]
        
        # Reverse diffusion
        solver_fn = self.heun_step if solver == 'heun' else self.euler_maruyama_step
        
        for i in tqdm(range(num_steps), desc='Sampling'):
            t = timesteps[i].expand(batch_size * num_samples)
            x = solver_fn(x, t, dt, history_flat, guidance_scale)
            
            if return_all_steps:
                all_steps.append(x.clone())
        
        # Reshape results
        x = x.reshape(batch_size, num_samples, predict_len, features)
        
        if return_all_steps:
            all_steps = [step.reshape(batch_size, num_samples, predict_len, features) 
                        for step in all_steps]
            return all_steps
        
        return x
    
    def predict(
        self,
        history: Union[torch.Tensor, np.ndarray],
        num_samples: int = 100,
        num_steps: int = 1000,
        solver: str = 'euler',
        guidance_scale: float = 1.0,
        denormalize: bool = True,
        return_dict: bool = True
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Generate predictions with optional denormalization.
        
        Args:
            history: Historical data
            num_samples: Number of prediction samples
            num_steps: Number of diffusion steps
            solver: Solver type
            guidance_scale: Guidance scale
            denormalize: Whether to denormalize predictions
            return_dict: Whether to return dictionary with statistics
            
        Returns:
            Predictions array or dictionary with statistics
        """
        # Convert to tensor
        if isinstance(history, np.ndarray):
            history = torch.from_numpy(history).float()
        
        # Add batch dimension if needed
        if history.dim() == 2:
            history = history.unsqueeze(0)
        
        history = history.to(self.device)
        
        # Generate samples
        with torch.no_grad():
            samples = self.sample_trajectories(
                history=history,
                num_samples=num_samples,
                num_steps=num_steps,
                solver=solver,
                guidance_scale=guidance_scale
            )
        
        # Move to CPU
        samples = samples.cpu()
        
        # Denormalize if requested
        if denormalize and self.preprocessor is not None:
            # Reshape for denormalization
            original_shape = samples.shape
            samples_flat = samples.reshape(-1, samples.shape[-1])
            samples_flat = self.preprocessor.denormalize(samples_flat)
            samples = samples_flat.reshape(original_shape)
        
        # Convert to numpy
        samples_np = samples.numpy()
        
        if not return_dict:
            return samples_np
        
        # Compute statistics
        return {
            'samples': samples_np,
            'mean': samples_np.mean(axis=1),
            'std': samples_np.std(axis=1),
            'quantiles': {
                q: np.quantile(samples_np, q, axis=1)
                for q in [0.05, 0.25, 0.5, 0.75, 0.95]
            }
        }
    
    def predict_rolling(
        self,
        initial_history: torch.Tensor,
        num_windows: int,
        num_samples: int = 100,
        **kwargs
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate rolling predictions, using previous predictions as history.
        
        Args:
            initial_history: Initial historical window
            num_windows: Number of prediction windows
            num_samples: Samples per window
            **kwargs: Additional arguments for predict()
            
        Returns:
            List of prediction dictionaries for each window
        """
        predictions = []
        current_history = initial_history.clone()
        
        for i in range(num_windows):
            # Generate predictions
            pred = self.predict(
                current_history,
                num_samples=num_samples,
                return_dict=True,
                **kwargs
            )
            predictions.append(pred)
            
            # Update history with mean prediction
            mean_pred = torch.from_numpy(pred['mean']).to(current_history.device)
            
            # Shift history and append prediction
            if mean_pred.dim() == 2:
                mean_pred = mean_pred.unsqueeze(0)
            
            current_history = torch.cat([
                current_history[:, mean_pred.shape[1]:],
                mean_pred
            ], dim=1)
        
        return predictions