"""
Training logic for Bitcoin diffusion models.

Implements the training loop with score matching objective,
learning rate scheduling, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import json
from copy import deepcopy

from ..models import DiffusionTransformer, create_model
from ..models.sde import create_sde, SDE
from ..data import SlidingWindowDataset


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
    
    def update(self):
        """Update shadow parameters."""
        with torch.no_grad():
            for param, shadow_param in zip(self.model.parameters(), self.shadow.parameters()):
                shadow_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        with torch.no_grad():
            for param, shadow_param in zip(self.model.parameters(), self.shadow.parameters()):
                param.data.copy_(shadow_param.data)
    
    def restore(self):
        """Restore original parameters."""
        with torch.no_grad():
            for param, shadow_param in zip(self.model.parameters(), self.shadow.parameters()):
                shadow_param.data.copy_(param.data)


class DiffusionTrainer:
    """Trainer for diffusion models."""
    
    def __init__(
        self,
        model: nn.Module,
        sde: SDE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            model: Diffusion model to train
            sde: SDE defining the diffusion process
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
        """
        self.model = model
        self.sde = sde
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        
        # Device with MPS support for MacBook
        device_str = config.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)
        
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if hasattr(torch.backends, 'mps'):
            print(f"MPS available: {torch.backends.mps.is_available()}")
        else:
            print("MPS not available (PyTorch version < 1.12)")
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            betas=config.get('betas', (0.9, 0.999)),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100) * len(train_loader),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # EMA
        self.ema = None
        if config.get('use_ema', True):
            self.ema = EMA(self.model, decay=config.get('ema_decay', 0.999))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Loss weighting
        self.loss_weight_type = config.get('loss_weight_type', 'uniform')
        
        # Gradient clipping
        self.gradient_clip = config.get('gradient_clip', 1.0)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the score matching loss with classifier-free guidance.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor and metrics dictionary
        """
        # Extract data
        x_history = batch['history'].to(self.device)
        x_future = batch['future'].to(self.device)
        batch_size = x_history.shape[0]
        
        # Sample random timesteps
        t = torch.rand(batch_size, device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_future)
        
        # Get noised future values using SDE
        x_t = self.sde.sample_transition(x_future, t, noise)
        
        # Classifier-free guidance: randomly drop conditioning
        cond_drop_mask = None
        if self.model.training and hasattr(self.model, 'cond_drop_prob'):
            # Get the drop probability from model
            drop_prob = self.model.cond_drop_prob
            if drop_prob > 0:
                # Create binary mask: 1 = keep conditioning, 0 = drop conditioning
                cond_drop_mask = torch.bernoulli(
                    torch.ones(batch_size, device=self.device) * (1 - drop_prob)
                )
        
        # Predict score with potential conditioning dropout
        score_pred = self.model(x_t, x_history, t, cond_drop_mask)
        
        # Compute target for score-based diffusion: ||sigma_t * score + z||^2
        mean, std = self.sde.transition_kernel(x_future, t)
        # Standard formulation: model predicts score S(x,t), target is -z
        std_expanded = std.view(-1, *([1] * (noise.dim() - 1)))
        score_target = -noise  # Target is negative of sampled z ~ N(0,1)
        
        # Apply the score-based loss: ||sigma_t * score_pred + z||^2
        # This is equivalent to ||score_pred - (-z/sigma_t)||^2 * sigma_t^2
        scaled_pred = std_expanded * score_pred
        scaled_target = -noise  # This is -z
        
        # Loss weighting for stability
        if self.loss_weight_type == 'uniform':
            weight = 1.0
        elif self.loss_weight_type == 'importance':
            # Weight by inverse variance for importance sampling
            weight = 1.0 / (std_expanded ** 2 + 1e-8)
        elif self.loss_weight_type == 'likelihood':
            # Weight by g(t)^2 for likelihood weighting
            g_t = self.sde.g(t).view(-1, *([1] * (noise.dim() - 1)))
            weight = g_t ** 2
        else:
            weight = 1.0
        
        # Compute the score-based loss: ||sigma_t * score + z||^2
        mse_loss = (scaled_pred + noise) ** 2
        
        # Apply loss clipping to prevent exploding losses
        mse_loss = torch.clamp(mse_loss, max=100.0)  # Clip very large losses
        
        loss = torch.mean(weight * mse_loss)
        
        # Compute metrics
        with torch.no_grad():
            # Check for problematic values
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is {loss.item()}, skipping batch")
                return torch.tensor(0.0, device=self.device, requires_grad=True), {'loss': 0.0}
            
            metrics = {
                'loss': loss.item(),
                'score_pred_norm': score_pred.norm(dim=-1).mean().item(),
                'score_target_norm': score_target.norm(dim=-1).mean().item(),
                'score_pred_max': score_pred.abs().max().item(),
                'score_target_max': score_target.abs().max().item(),
                'noise_level_mean': std.mean().item(),
                'noise_level_max': std.max().item(),
                'x_t_norm': x_t.norm(dim=-1).mean().item(),
                'x_future_norm': x_future.norm(dim=-1).mean().item(),
                't_mean': t.mean().item(),
            }
            
            # Add CFG metrics if applicable
            if cond_drop_mask is not None:
                metrics['cond_drop_rate'] = (1 - cond_drop_mask.mean()).item()
            
            # Add weight statistics
            if hasattr(weight, 'mean'):
                metrics['loss_weight_mean'] = weight.mean().item()
                metrics['loss_weight_max'] = weight.max().item()
        
        return loss, metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        for batch in pbar:
            # Compute loss
            loss, metrics = self.compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # EMA update
            if self.ema is not None:
                self.ema.update()
            
            # Update metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
            
            self.global_step += 1
        
        # Average metrics
        epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        epoch_metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_metrics = {}
        
        # Use EMA weights for validation
        if self.ema is not None:
            self.ema.apply_shadow()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                loss, metrics = self.compute_loss(batch)
                
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(v)
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        # Average metrics
        val_metrics = {f'val_{k}': np.mean(v) for k, v in val_metrics.items()}
        
        return val_metrics
    
    def save_checkpoint(self, filename: str = 'checkpoint.pt', is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.shadow.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, self.checkpoint_dir / filename)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.shadow.load_state_dict(checkpoint['ema_state_dict'])
    
    def train(self, num_epochs: int):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        # Training history
        history = {'train': {}, 'val': {}}
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            for k, v in all_metrics.items():
                split = 'val' if k.startswith('val_') else 'train'
                metric_name = k.replace('val_', '') if split == 'val' else k
                
                if metric_name not in history[split]:
                    history[split][metric_name] = []
                history[split][metric_name].append(v)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics.get('val_loss', 'N/A')}")
            
            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pt')
            
            # Check if best model
            if 'val_loss' in val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(is_best=True)
                print(f"New best model! Val Loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history


def create_trainer(config: Dict) -> DiffusionTrainer:
    """
    Factory function to create a trainer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured trainer instance
    """
    # Create model
    model = create_model(config['model'])
    
    # Create SDE
    sde = create_sde(**config['sde'])
    
    # Create data loaders
    from ..data import SlidingWindowDataset, DataPreprocessor
    
    # Load processed data
    preprocessor = DataPreprocessor(config['data'])
    train_data, _ = preprocessor.load_processed_data(config['data']['train_path'])
    val_data, _ = preprocessor.load_processed_data(config['data']['val_path'])
    
    # Create datasets
    train_dataset = SlidingWindowDataset(
        train_data,
        history_len=config['model']['history_len'],
        predict_len=config['model']['predict_len'],
        **config['data'].get('dataset', {})
    )
    
    val_dataset = SlidingWindowDataset(
        val_data,
        history_len=config['model']['history_len'],
        predict_len=config['model']['predict_len'],
        **config['data'].get('dataset', {})
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        sde=sde,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training']
    )
    
    return trainer