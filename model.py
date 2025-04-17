import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(1, dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        """
        Args:
            t: [B, 1]
        Returns:
            [B, dim]
        """
        t = self.linear1(t)
        t = self.act(t)
        t = self.linear2(t)
        return t  # [B, dim]

class ScoreTransformerNet(nn.Module):
    def __init__(self, input_dim=1, history_len=50, predict_len=50,
                 model_dim=256, num_heads=4, num_layers=4):
        super().__init__()

        self.history_len = history_len
        self.predict_len = predict_len
        self.total_len = history_len + predict_len

        self.input_proj = nn.Linear(input_dim, model_dim)
        self.time_embed = TimeEmbedding(model_dim)

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
            x_t:      [B, predict_len, input_dim]  ← noised future
            x_history: [B, history_len, input_dim] ← conditioning history
            t:        [B, 1] ← diffusion time
        Returns:
            score_pred: [B, predict_len, input_dim]
        """
        B = x_t.size(0)
        x_combined = torch.cat([x_history, x_t], dim=1)           # [B, total_len, input_dim]
        x_embed = self.input_proj(x_combined)                     # [B, total_len, model_dim]

        t_embed = self.time_embed(t).unsqueeze(1).expand(-1, self.total_len, -1)  # [B, total_len, model_dim]
        tokens = x_embed + t_embed                                # [B, total_len, model_dim]

        encoded = self.transformer(tokens)                        # [B, total_len, model_dim]
        future_encoded = encoded[:, self.history_len:, :]        # [B, predict_len, model_dim]

        score = self.output_proj(future_encoded)                 # [B, predict_len, input_dim]
        return score                                              # ✅ correct: [B, predict_len, 1]
