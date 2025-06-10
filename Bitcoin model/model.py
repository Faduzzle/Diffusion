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

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_embed = TimeEmbedding(model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        self.concat_fuse = nn.Linear(3 * model_dim, model_dim)

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

    def forward(self, x_t, x_history, t, cond_drop_prob=0.0):
        B, _, _ = x_t.shape
        device = x_t.device

        if self.training and cond_drop_prob > 0.0:
            mask = (torch.rand(B, device=device) > cond_drop_prob).float().view(B, 1, 1)
            x_history = x_history * mask

        t_embed = self.time_embed(t)
        t_future = t_embed.unsqueeze(1).expand(-1, self.predict_len, -1)
        t_hist = t_embed.unsqueeze(1).expand(-1, self.history_len, -1)

        pe_future = self.pos_encoding(self.predict_len, device).expand(B, -1, -1)
        pe_hist = self.pos_encoding(self.history_len, device).expand(B, -1, -1)

        x_future = self.input_proj(x_t)
        x_hist = self.input_proj(x_history)

        x_future = torch.cat([x_future, t_future, pe_future], dim=-1)
        x_hist = torch.cat([x_hist, t_hist, pe_hist], dim=-1)

        x_future = self.concat_fuse(x_future)
        x_hist = self.concat_fuse(x_hist)

        h_encoded = self.encoder(x_hist)
        decoded = self.decoder(x_future, h_encoded)

        return self.output_proj(decoded)
