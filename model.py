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
        return t

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        out, _ = self.attn(query=query, key=key, value=value)
        return out

class ScoreTransformerNet(nn.Module):
    def __init__(self, input_dim, history_len, predict_len, model_dim=256, num_heads=4, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        self.history_len = history_len
        self.predict_len = predict_len
        self.total_len = history_len + predict_len

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_embed = TimeEmbedding(model_dim)
        self.pos_encoding = LearnablePositionalEncoding(model_dim, max_len=self.total_len)
        self.token_type_embed = nn.Embedding(2, model_dim)
        self.cross_attn = CrossAttention(model_dim, num_heads)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x_t, x_history, t):
        B = x_history.size(0)

        # Project inputs
        h_embed = self.input_proj(x_history)         # [B, history_len, D]
        x_embed = self.input_proj(x_t)               # [B, predict_len, D]

        # Positional Encoding
        h_embed = self.pos_encoding(h_embed)
        x_embed = self.pos_encoding(x_embed)

        # Time Embedding (only to future)
        t_embed = self.time_embed(t).unsqueeze(1).expand(-1, self.predict_len, -1)
        x_embed = x_embed + t_embed

        # Token Type Embedding
        h_type_ids = torch.zeros(B, self.history_len, dtype=torch.long, device=x_history.device)
        x_type_ids = torch.ones(B, self.predict_len, dtype=torch.long, device=x_t.device)
        h_embed = h_embed + self.token_type_embed(h_type_ids)
        x_embed = x_embed + self.token_type_embed(x_type_ids)

        # Cross Attention: future attends to history
        x_embed = x_embed + self.cross_attn(x_embed, h_embed, h_embed)

        # Concatenate and pass through transformer
        tokens = torch.cat([h_embed, x_embed], dim=1)
        tokens = self.transformer(tokens)

        # Take only future tokens
        future_encoded = tokens[:, self.history_len:, :]
        score_pred = self.output_proj(future_encoded)

        return score_pred
