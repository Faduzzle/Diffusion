import torch
import torch.nn as nn

class ConditionalScoreNet(nn.Module):
    def __init__(self, input_dim=1, history_len=150, predict_len=50, model_dim=256, num_heads=4, num_layers=4):
        super().__init__()

        self.history_len = history_len
        self.predict_len = predict_len
        self.total_len = history_len + predict_len

        self.input_proj = nn.Linear(input_dim, model_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )

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
        x_t:        (B, predict_len, input_dim)   — noised future
        x_history:  (B, history_len, input_dim)   — clean past
        t:          (B, 1)                         — diffusion time
        """
        B = x_t.size(0)

        # Combine history + future
        x_combined = torch.cat([x_history, x_t], dim=1)      # (B, total_len, input_dim)
        x_embed = self.input_proj(x_combined)                # (B, total_len, model_dim)

        # Broadcast time embedding
        t_embed = self.time_mlp(t)                           # (B, model_dim)
        t_embed = t_embed.unsqueeze(1).expand(-1, self.total_len, -1)

        # Add time embedding to all tokens
        tokens = x_embed + t_embed                           # (B, total_len, model_dim)

        # Encode with transformer
        encoded = self.transformer(tokens)                   # (B, total_len, model_dim)

        # Extract the future half
        future_encoded = encoded[:, self.history_len:, :]    # (B, predict_len, model_dim)

        return self.output_proj(future_encoded)              # (B, predict_len, input_dim)
