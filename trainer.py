import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import ScoreTransformerNet
from sde import VPSDE
from data import load_csv_time_series
from config import CONFIG


def conditional_score_matching_loss(score_pred, x_t, x_0, alpha_t, sigma_t, eps=1e-5):
    target_score = -(x_t - alpha_t * x_0) / (sigma_t + eps)
    return nn.functional.mse_loss(score_pred, target_score)


class DiffusionTrainer:
    def __init__(self, config):
        self.device = torch.device(config["device"])
        self.history_len = config["history_len"]
        self.predict_len = config["predict_len"]
        self.input_dim = config["input_dim"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.lr = config["lr"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.save_name = config["save_name"]
        self.num_diffusion_timesteps = config["num_diffusion_timesteps"]

        # Load dataset
        self.train_data, self.norm_factor = load_csv_time_series(
            csv_path=config["train_csv_path"],
            history_len=self.history_len,
            predict_len=self.predict_len,
        )
        self.train_data = self.train_data.to(self.device)

        # Initialize model and optimizer
        self.model = ScoreTransformerNet(
            input_dim=self.input_dim,
            history_len=self.history_len,
            predict_len=self.predict_len,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.sde = VPSDE()

    def train(self):
        for epoch in range(self.n_epochs):
            self.model.train()

            indices = torch.randint(0, self.train_data.shape[0], (self.batch_size,))
            batch = self.train_data[indices]  # [B, T, input_dim]
            x_history = batch[:, :self.history_len, :]
            x_future_clean = batch[:, self.history_len:, :]

            t = torch.rand(self.batch_size, 1, device=self.device)  # [B, 1]
            alpha_t = self.sde.alpha(t)  # [B, predict_len, 1]
            sigma_t = torch.sqrt(1.0 - alpha_t ** 2)  # [B, predict_len, 1]

            noise = torch.randn_like(x_future_clean)
            x_t = alpha_t * x_future_clean + sigma_t * noise

            score_pred = self.model(x_t, x_history, t)
            loss = conditional_score_matching_loss(score_pred, x_t, x_future_clean, alpha_t, sigma_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} | Loss: {loss.item():.6f}")

            if (epoch + 1) % CONFIG["checkpoint_freq"] == 0 or epoch + 1 == self.n_epochs:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        save_dict = {
            "epoch": epoch,
            "score_net_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "norm_factor": self.norm_factor,
        }
        save_path = f"{self.checkpoint_dir}/{self.save_name}_epoch{epoch}.pth"
        torch.save(save_dict, save_path)
        print(f"✅ Saved checkpoint to {save_path}")
