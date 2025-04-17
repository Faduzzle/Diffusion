import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        t = self.linear1(t)
        t = self.act(t)
        t = self.linear2(t)
        return t

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ScoreTransformer(nn.Module):
    def __init__(self, hist_len, pred_len, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.total_len = hist_len + pred_len
        self.pred_len = pred_len
        self.input_proj = nn.Linear(1, d_model)
        self.time_embed = TimeEmbedding(d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len=self.total_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x_t, t, history):
        batch_size = x_t.size(0)
        full_seq = torch.cat([history, x_t], dim=1)
        x = self.input_proj(full_seq)
        t_embed = self.time_embed(t).unsqueeze(1)
        x = self.pos_embed(x) + t_embed
        encoded = self.transformer(x)
        pred_repr = encoded[:, -self.pred_len:]
        score = self.output_proj(pred_repr)
        return score.squeeze(-1)
