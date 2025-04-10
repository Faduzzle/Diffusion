import torch
import torch.nn as nn

class ConditionalScoreNet(nn.Module):
    def __init__(self, input_dim=1, history_len=80, predict_len=50, model_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=4 * model_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.history_proj = nn.Linear(input_dim, model_dim)
        self.predict_proj = nn.Linear(input_dim, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        self.out = nn.Linear(model_dim, input_dim)
        self.history_len = history_len
        self.predict_len = predict_len

    def forward(self, x_t, x_history, t):
        batch_size = x_t.shape[0]
        h = self.history_proj(x_history)
        context = self.history_encoder(h)
        context = context.mean(dim=1)

        t_embed = self.time_mlp(t)
        x_proj = self.predict_proj(x_t)
        context = context.unsqueeze(1).expand(-1, self.predict_len, -1)
        t_embed = t_embed.unsqueeze(1).expand(-1, self.predict_len, -1)

        z = x_proj + context + t_embed      # run each one with their own encoding
        return self.out(z)