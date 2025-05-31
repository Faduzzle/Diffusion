"""
Bitcoin Price Diffusion Model using Transformer Architecture

This module implements a score-based diffusion model for time series prediction,
specifically designed for Bitcoin price modeling. It uses a Transformer encoder-decoder
architecture to learn the score function for the reverse diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """Embeds diffusion timesteps into a high-dimensional representation."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps tensor of shape (batch_size,)
        Returns:
            Time embeddings of shape (batch_size, dim)
        """
        # Reshape to (batch_size, 1) for linear layer
        t = t.unsqueeze(-1).float()
        return self.mlp(t)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create a buffer for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sinusoidal encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encoding of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class DiffusionTransformer(nn.Module):
    """
    Transformer-based score network for diffusion models.
    
    Uses an encoder-decoder architecture where:
    - Encoder processes historical data for context
    - Decoder predicts scores for the noised future values
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        history_len: int = 252,  # ~1 year of trading days
        predict_len: int = 21,   # ~1 month ahead
        dropout: float = 0.1,
        cond_drop_prob: float = 0.1,  # For classifier-free guidance
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.history_len = history_len
        self.predict_len = predict_len
        self.cond_drop_prob = cond_drop_prob
        
        # Time embedding
        self.time_embed = TimeEmbedding(model_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, max_len=history_len + predict_len)
        
        # Input projections
        self.history_proj = nn.Linear(input_dim, model_dim)
        self.future_proj = nn.Linear(input_dim, model_dim)
        
        # Transformer encoder for historical context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer decoder for score prediction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(model_dim, input_dim)
        
        # Optional: Variance prediction head
        self.predict_variance = False
        if self.predict_variance:
            self.var_proj = nn.Linear(model_dim, input_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x_t: torch.Tensor, 
        x_history: torch.Tensor, 
        t: torch.Tensor,
        cond_drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion transformer.
        
        Args:
            x_t: Noised future values, shape (batch_size, predict_len, input_dim)
            x_history: Historical context, shape (batch_size, history_len, input_dim)
            t: Diffusion timesteps, shape (batch_size,)
            cond_drop_mask: Optional mask for classifier-free guidance
            
        Returns:
            Predicted score (and optionally variance)
        """
        batch_size = x_t.size(0)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch_size, model_dim)
        
        # Project inputs to model dimension
        history_emb = self.history_proj(x_history)  # (batch_size, history_len, model_dim)
        future_emb = self.future_proj(x_t)  # (batch_size, predict_len, model_dim)
        
        # Add positional encoding
        history_pos = self.pos_encoder(history_emb)
        future_pos = self.pos_encoder.pe[:, self.history_len:self.history_len + self.predict_len, :]
        
        history_emb = history_emb + history_pos
        future_emb = future_emb + future_pos.expand(batch_size, -1, -1)
        
        # Apply dropout
        history_emb = self.dropout(history_emb)
        future_emb = self.dropout(future_emb)
        
        # Classifier-free guidance: conditionally drop history
        if cond_drop_mask is not None:
            # cond_drop_mask is 1 for conditional, 0 for unconditional
            # We zero out history embeddings for unconditional samples
            history_emb = history_emb * cond_drop_mask.view(-1, 1, 1)
        
        memory = self.encoder(history_emb)
        
        # Add time embedding to future embeddings
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, self.predict_len, -1)
        future_emb = future_emb + t_emb_expanded
        
        # Decode to predict scores
        decoder_out = self.decoder(
            tgt=future_emb,
            memory=memory
        )
        
        # Apply layer norm
        decoder_out = self.layer_norm(decoder_out)
        
        # Project to output dimension
        score = self.output_proj(decoder_out)
        
        if self.predict_variance:
            variance = F.softplus(self.var_proj(decoder_out)) + 1e-3
            return score, variance
        
        return score
    
    def get_uncond_score(self, x_t: torch.Tensor, x_history: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get unconditional score for classifier-free guidance.
        
        This passes a zero mask to drop all conditioning information,
        making the model predict as if no historical context was provided.
        """
        batch_size = x_t.size(0)
        # Create mask of zeros to drop all conditioning
        cond_drop_mask = torch.zeros(batch_size, device=x_t.device)
        return self.forward(x_t, x_history, t, cond_drop_mask)
    
    def get_cond_score(self, x_t: torch.Tensor, x_history: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Get conditional score (standard forward pass).
        
        This uses full historical conditioning.
        """
        batch_size = x_t.size(0)
        # Create mask of ones to keep all conditioning
        cond_drop_mask = torch.ones(batch_size, device=x_t.device)
        return self.forward(x_t, x_history, t, cond_drop_mask)


class LatentDiffusionModel(nn.Module):
    """
    Optional wrapper for latent diffusion modeling.
    
    Encodes time series to a latent space before diffusion,
    which can be beneficial for high-dimensional or complex data.
    """
    
    def __init__(
        self,
        score_model: DiffusionTransformer,
        latent_dim: int = 64,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
    ):
        super().__init__()
        
        self.score_model = score_model
        self.latent_dim = latent_dim
        
        # Encoder: Maps time series to latent space
        encoder_channels = [score_model.input_dim, 32, 64, latent_dim]
        encoder_modules = []
        
        for i in range(len(encoder_channels) - 1):
            encoder_modules.extend([
                nn.Conv1d(encoder_channels[i], encoder_channels[i+1], 
                         kernel_size=3, padding=1),
                nn.GroupNorm(8, encoder_channels[i+1]),
                nn.GELU()
            ])
        
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Decoder: Maps latent space back to time series
        decoder_channels = list(reversed(encoder_channels))
        decoder_modules = []
        
        for i in range(len(decoder_channels) - 1):
            decoder_modules.extend([
                nn.ConvTranspose1d(decoder_channels[i], decoder_channels[i+1],
                                  kernel_size=3, padding=1),
                nn.GroupNorm(8, decoder_channels[i+1]) if i < len(decoder_channels) - 2 else nn.Identity(),
                nn.GELU() if i < len(decoder_channels) - 2 else nn.Identity()
            ])
        
        self.decoder = nn.Sequential(*decoder_modules)
        
        # Update score model dimensions
        self.score_model.input_dim = latent_dim
        self.score_model.history_proj = nn.Linear(latent_dim, score_model.model_dim)
        self.score_model.future_proj = nn.Linear(latent_dim, score_model.model_dim)
        self.score_model.output_proj = nn.Linear(score_model.model_dim, latent_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time series to latent space."""
        # x: (batch_size, seq_len, channels) -> (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        # z: (batch_size, latent_dim, seq_len) -> (batch_size, seq_len, latent_dim)
        return z.transpose(1, 2)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to time series."""
        # z: (batch_size, seq_len, latent_dim) -> (batch_size, latent_dim, seq_len)
        z = z.transpose(1, 2)
        x = self.decoder(z)
        # x: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        return x.transpose(1, 2)
    
    def forward(
        self,
        x_t: torch.Tensor,
        x_history: torch.Tensor,
        t: torch.Tensor,
        cond_drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through latent diffusion model."""
        # Encode to latent space
        z_t = self.encode(x_t)
        z_history = self.encode(x_history)
        
        # Get score in latent space
        score_z = self.score_model(z_t, z_history, t, cond_drop_mask)
        
        # Decode score back to data space
        score = self.decode(score_z)
        
        return score


def create_model(config: dict, use_latent: bool = False) -> nn.Module:
    """
    Factory function to create a diffusion model.
    
    Args:
        config: Configuration dictionary
        use_latent: Whether to use latent diffusion
        
    Returns:
        Configured model instance
    """
    # Create base transformer model
    transformer = DiffusionTransformer(
        input_dim=config.get('input_dim', 1),
        model_dim=config.get('model_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        ff_dim=config.get('ff_dim', 1024),
        history_len=config.get('history_len', 252),
        predict_len=config.get('predict_len', 21),
        dropout=config.get('dropout', 0.1),
        cond_drop_prob=config.get('cond_drop_prob', 0.1),
    )
    
    if use_latent:
        return LatentDiffusionModel(
            score_model=transformer,
            latent_dim=config.get('latent_dim', 64),
            encoder_layers=config.get('encoder_layers', 3),
            decoder_layers=config.get('decoder_layers', 3),
        )
    
    return transformer