import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ScoreTransformerNet
from sde import VPSDE
from data import generate_sine_sequence

# ==== CONFIGURATION ====
CONFIG = {
    "checkpoint_path": "checkpoints\latest.pth",  # <--- change this path to your model
    "history_len": 50,
    "predict_len": 150,
    "input_dim": 1,
    "num_samples": 5,
    "num_diffusion_timesteps": 1000
}


def predict_and_save_inline(checkpoint_path, history_len=50, predict_len=150, input_dim=1,
                            num_samples=5, num_diffusion_timesteps=1000):

    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ScoreTransformerNet(input_dim, history_len, predict_len).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["score_net_state_dict"])
    model.eval()

    # Setup SDE
    sde = VPSDE()

    # Generate sine data
    data = generate_sine_sequence(num_samples, history_len + predict_len).to(device)
    x_history = data[:, :history_len, :]
    x_true = data[:, history_len:, :]

    # Initial noise for future
    x = torch.randn((num_samples, predict_len, input_dim), device=device)

    # Output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "Data", "Predictions")
    os.makedirs(save_dir, exist_ok=True)

    # Reverse SDE sampling
    dt = -1.0 / num_diffusion_timesteps
    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1),
                      desc="Sampling", leave=True, dynamic_ncols=True):
            t_val = max(i / num_diffusion_timesteps, 1e-5)
            t = torch.full((num_samples, 1), t_val, device=device)
            score = model(x, x_history, t)
            drift = sde.f(x, t) - (sde.g(t) ** 2) * score
            z = torch.randn_like(x)
            x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z

    # Post-process results
    x_pred = x.cpu().squeeze(-1)
    x_hist = x_history.cpu().squeeze(-1)
    x_true = x_true.cpu().squeeze(-1)

    # Save to CSV
    full_seq = torch.cat([x_hist, x_pred], dim=1).numpy()
    columns = [f"h_{i}" for i in range(history_len)] + [f"p_{i}" for i in range(predict_len)]
    df = pd.DataFrame(full_seq, columns=columns)
    csv_path = os.path.join(save_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to: {csv_path}")

    # Save plots
    for i in range(min(5, num_samples)):
        plt.figure(figsize=(10, 4))
        plt.plot(range(history_len), x_hist[i], label="History")
        plt.plot(range(history_len, history_len + predict_len), x_true[i], label="True Future", linestyle="--")
        plt.plot(range(history_len, history_len + predict_len), x_pred[i], label="Predicted Future")
        plt.title(f"Sample {i}")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to: {plot_path}")

    print(f"✅ All outputs saved in: {save_dir}")


if __name__ == "__main__":
    predict_and_save_inline(**CONFIG)
