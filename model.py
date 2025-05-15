import torch
import torch.nn as nn
import math

# === Time Embedding ===
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

# === Positional Encoding ===
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

# === Time Series Encoder ===
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: [B, T, D] → [B, D, T] → conv → [B, L, T] → [B, T, L]
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)

# === Time Series Decoder ===
class TimeSeriesDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)
        )

    def forward(self, z):
        # z: [B, T, L] → [B, L, T] → conv → [B, D, T] → [B, T, D]
        return self.net(z.permute(0, 2, 1)).permute(0, 2, 1)

# === Cross Attention Block ===
class CrossAttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, query, context):
        attn_out, _ = self.attn(query, context, context)
        out = self.norm(query + attn_out)
        return out + self.ffn(out)

# === Transformer Score Network with Cross-Attention + Variance Head ===
class ScoreTransformerNet(nn.Module):
    def __init__(self, latent_dim=32, model_dim=256, num_heads=4, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, model_dim)
        self.time_embed = TimeEmbedding(model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.concat_fuse = nn.Linear(3 * model_dim, model_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                       dim_feedforward=4 * model_dim, batch_first=True),
            num_layers=num_layers
        )

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(model_dim, num_heads) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, latent_dim)
        self.output_var = nn.Linear(model_dim, latent_dim)

    def forward(self, z_t, z_history, t, cond_drop_prob=0.0):
        B, T_pred, _ = z_t.shape
        T_hist = z_history.shape[1]
        device = z_t.device

        if self.training and cond_drop_prob > 0.0:
            mask = (torch.rand(B, device=device) > cond_drop_prob).float().view(B, 1, 1)
            z_history = z_history * mask

        t_embed = self.time_embed(t)  # [B, model_dim]
        t_future = t_embed.unsqueeze(1).expand(-1, T_pred, -1)
        t_hist = t_embed.unsqueeze(1).expand(-1, T_hist, -1)

        pe_future = self.pos_encoding(T_pred, device).expand(B, -1, -1)
        pe_hist = self.pos_encoding(T_hist, device).expand(B, -1, -1)

        z_future = self.input_proj(z_t)
        z_hist = self.input_proj(z_history)

        z_future = torch.cat([z_future, t_future, pe_future], dim=-1)
        z_hist = torch.cat([z_hist, t_hist, pe_hist], dim=-1)

        z_future = self.concat_fuse(z_future)
        z_hist = self.concat_fuse(z_hist)

        encoded_history = self.encoder(z_hist)

        for block in self.cross_attn_blocks:
            z_future = block(z_future, encoded_history)

        out = self.norm(z_future)
        return self.output_proj(out), self.output_var(out)

# === Full Latent Diffusion Model ===
class LatentDiffusionModel(nn.Module):
    def __init__(self, input_dim, latent_dim=32, model_dim=256,
                 num_heads=4, num_layers=4):
        super().__init__()
        self.encoder = TimeSeriesEncoder(input_dim, latent_dim)
        self.decoder = TimeSeriesDecoder(latent_dim, input_dim)
        self.score_net = ScoreTransformerNet(latent_dim, model_dim, num_heads, num_layers)

    def forward(self, x_t, x_history, t, cond_drop_prob=0.0):
        z_t = self.encoder(x_t)
        z_history = self.encoder(x_history)
        score, variance = self.score_net(z_t, z_history, t, cond_drop_prob)
        return score, variance

    def decode(self, z):
        return self.decoder(z)
