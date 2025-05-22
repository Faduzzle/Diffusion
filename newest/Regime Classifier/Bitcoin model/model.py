import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.net(t)  # [B, dim]

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, length, device):
        position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(length, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

class ScoreTransformerNet(nn.Module):
    def __init__(self, input_dim=1, history_len=50, predict_len=20,
                 model_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.history_len = history_len
        self.predict_len = predict_len
        self.input_dim = input_dim
        self.model_dim = model_dim

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_embed = TimeEmbedding(model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.concat_fuse = nn.Linear(3 * model_dim, model_dim)

        # Initialize mask token with model dimension instead of input dimension
        self.mask_token = nn.Parameter(torch.randn(1, model_dim))
        
        # Add null conditional embedding for CFG
        self.null_cond = nn.Parameter(torch.randn(1, history_len, model_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                       dim_feedforward=4 * model_dim, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads,
                                       dim_feedforward=4 * model_dim, batch_first=True),
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(model_dim, input_dim)

    def apply_random_masks(self, x, mask_ratio=0.15):
        """Apply random token masking"""
        B, L, D = x.shape
        num_masks = int(L * mask_ratio)
        x_masked = x.clone()
        for i in range(B):
            mask_indices = torch.randperm(L)[:num_masks]
            # Cast mask token to match input dtype
            mask_token = self.mask_token.to(dtype=x.dtype)
            x_masked[i, mask_indices] = mask_token.expand(num_masks, -1)
        return x_masked

    def get_conditional_score(self, x_t, x_history, t, cond_drop_prob=0.1, mask_ratio=0.15):
        """Get score with conditional information (history)"""
        B, _, _ = x_t.shape
        device = x_t.device

        t_embed = self.time_embed(t)
        t_future = t_embed.unsqueeze(1).expand(-1, self.predict_len, -1)
        t_hist = t_embed.unsqueeze(1).expand(-1, self.history_len, -1)

        pe_future = self.pos_encoding(self.predict_len, device).expand(B, -1, -1)
        pe_hist = self.pos_encoding(self.history_len, device).expand(B, -1, -1)

        x_future = self.input_proj(x_t)
        x_hist = self.input_proj(x_history)

        # Apply random token masking during training
        if self.training:
            x_hist = self.apply_random_masks(x_hist, mask_ratio)
            x_future = self.apply_random_masks(x_future, mask_ratio)

        x_future = torch.cat([x_future, t_future, pe_future], dim=-1)
        x_hist = torch.cat([x_hist, t_hist, pe_hist], dim=-1)

        x_future = self.concat_fuse(x_future)
        x_hist = self.concat_fuse(x_hist)

        h_encoded = self.encoder(x_hist)
        decoded = self.decoder(x_future, h_encoded)

        return self.output_proj(decoded)

    def get_unconditional_score(self, x_t, t):
        """Get score without conditional information"""
        B, _, _ = x_t.shape
        device = x_t.device

        t_embed = self.time_embed(t)
        t_future = t_embed.unsqueeze(1).expand(-1, self.predict_len, -1)
        t_hist = t_embed.unsqueeze(1).expand(-1, self.history_len, -1)

        pe_future = self.pos_encoding(self.predict_len, device).expand(B, -1, -1)
        pe_hist = self.pos_encoding(self.history_len, device).expand(B, -1, -1)

        x_future = self.input_proj(x_t)
        null_hist = self.null_cond.expand(B, -1, -1)  # Use learned null conditional

        x_future = torch.cat([x_future, t_future, pe_future], dim=-1)
        null_hist = torch.cat([null_hist, t_hist, pe_hist], dim=-1)

        x_future = self.concat_fuse(x_future)
        null_hist = self.concat_fuse(null_hist)

        h_encoded = self.encoder(null_hist)
        decoded = self.decoder(x_future, h_encoded)

        return self.output_proj(decoded)

    def forward(self, x_t, x_history, t, cond_drop_prob=0.1, mask_ratio=0.15, cfg_weight=3.0):
        B, _, _ = x_t.shape
        device = x_t.device

        # During training, randomly choose between conditional and unconditional
        if self.training:
            use_cond = torch.rand(B, device=device) > cond_drop_prob
            scores = []
            for i in range(B):
                if use_cond[i]:
                    score = self.get_conditional_score(
                        x_t[i:i+1], 
                        x_history[i:i+1], 
                        t[i:i+1], 
                        mask_ratio=mask_ratio
                    )
                else:
                    score = self.get_unconditional_score(x_t[i:i+1], t[i:i+1])
                scores.append(score)
            return torch.cat(scores, dim=0)
        
        # During inference, use classifier-free guidance
        else:
            cond_score = self.get_conditional_score(x_t, x_history, t, mask_ratio=mask_ratio)
            uncond_score = self.get_unconditional_score(x_t, t)
            return uncond_score + cfg_weight * (cond_score - uncond_score)
