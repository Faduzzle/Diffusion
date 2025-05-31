"""
Transformer architecture for time series diffusion.

This is a port of the existing Bitcoin diffusion transformer to the new framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from ...core import BaseArchitecture, Registry


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
        # Reshape to (batch_size, 1) for linear layer
        t = t.unsqueeze(-1).float()
        return self.mlp(t)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


@Registry.register("architecture", "transformer")
class TransformerArchitecture(BaseArchitecture):
    """
    Transformer-based architecture for time series diffusion.
    
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
        history_len: int = 252,
        predict_len: int = 21,
        dropout: float = 0.1,
        cond_drop_prob: float = 0.1,
        attention_type: str = "full",
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            model_dim=model_dim,
            history_len=history_len,
            predict_len=predict_len,
            **kwargs
        )
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.cond_drop_prob = cond_drop_prob
        self.attention_type = attention_type
        
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
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_t: torch.Tensor,
        x_history: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            x_t: Noised future values (batch_size, predict_len, input_dim)
            x_history: Historical context (batch_size, history_len, input_dim)
            t: Diffusion timesteps (batch_size,)
            condition: Optional conditioning tensor or dropout mask
            
        Returns:
            Predicted score (batch_size, predict_len, input_dim)
        """
        batch_size = x_t.size(0)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch_size, model_dim)
        
        # Project inputs to model dimension
        history_emb = self.history_proj(x_history)
        future_emb = self.future_proj(x_t)
        
        # Add positional encoding
        history_pos = self.pos_encoder(history_emb)
        future_pos = self.pos_encoder.pe[:, self.history_len:self.history_len + self.predict_len, :]
        
        history_emb = history_emb + history_pos
        future_emb = future_emb + future_pos.expand(batch_size, -1, -1)
        
        # Apply dropout
        history_emb = self.dropout(history_emb)
        future_emb = self.dropout(future_emb)
        
        # Handle conditioning/dropout
        if condition is not None:
            # If condition is a binary mask for classifier-free guidance
            if condition.dim() == 1 and condition.shape[0] == batch_size:
                # condition is 1 for conditional, 0 for unconditional
                history_emb = history_emb * condition.view(-1, 1, 1)
            else:
                # condition is additional conditioning information
                # You can implement cross-attention or other conditioning methods here
                pass
        
        # Encode history
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
        
        return score
    
    def get_uncond_score(self, x_t: torch.Tensor, x_history: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get unconditional score for classifier-free guidance."""
        batch_size = x_t.size(0)
        cond_drop_mask = torch.zeros(batch_size, device=x_t.device)
        return self.forward(x_t, x_history, t, cond_drop_mask)
    
    def get_cond_score(self, x_t: torch.Tensor, x_history: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get conditional score (standard forward pass)."""
        batch_size = x_t.size(0)
        cond_drop_mask = torch.ones(batch_size, device=x_t.device)
        return self.forward(x_t, x_history, t, cond_drop_mask)