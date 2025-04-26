import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        t = self.linear1(t)
        t = self.act(t)
        t = self.linear2(t)
        return t  # [B, dim]

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # [B, T, dim]

class ScoreTransformerNet(nn.Module):
    def __init__(self, input_dim=1, history_len=50, predict_len=50,
                 model_dim=256, num_heads=4, num_layers=4):
        super().__init__()

        self.history_len = history_len
        self.predict_len = predict_len
        self.total_len = history_len + predict_len

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_embed = TimeEmbedding(model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_len=self.total_len)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=4 * model_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x_t, x_history, t):
        """
        Args:
            x_t: [B, predict_len, input_dim] — noised future
            x_history: [B, history_len, input_dim] — clean context
            t: [B, 1] — diffusion timestep
        Returns:
            score_pred: [B, predict_len, input_dim]
        """
        h_embed = self.input_proj(x_history)                     # [B, hist_len, model_dim]
        x_embed = self.input_proj(x_t)                           # [B, pred_len, model_dim]

        t_embed = self.time_embed(t).unsqueeze(1).expand(-1, self.predict_len, -1)
        x_embed = x_embed + t_embed                              # add t only to future

        tokens = torch.cat([h_embed, x_embed], dim=1)           # [B, total_len, model_dim]
        tokens = self.pos_encoding(tokens)                      # add positional encodings

        encoded = self.transformer(tokens)                      # [B, total_len, model_dim]
        future_encoded = encoded[:, self.history_len:, :]       # [B, pred_len, model_dim]

        score = self.output_proj(future_encoded)                # [B, pred_len, input_dim]
        return score
