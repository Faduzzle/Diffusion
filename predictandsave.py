import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ScoreTransformerNet
from sde import VPSDE
from data import load_tsf, slice_tsf_tensor, generate_sine_sequence, compute_all_metrics

# ==== CONFIGURATION ====
CONFIG = {
    "checkpoint_path": r"checkpoints\\latest.pth",
    "history_len": 50,
    "predict_len": 50,
    "input_dim": 1,
    "num_samples": 5,
    "num_diffusion_timesteps": 500,
    "num_paths": 1000,
    "use_tsf": True,  # set to True to use TSF file
    "tsf_path": "path/to/your_file.tsf"  # path to your TSF file
}

def predict_and_save_distribution(checkpoint_path, history_len=50, predict_len=50, input_dim=1,
                                   num_samples=5, num_diffusion_timesteps=1000, num_paths=1000,
                                   use_tsf=False, tsf_path=None):

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

    # Load data
    if use_tsf and tsf_path:
        full_tensor = load_tsf(tsf_path)
        data = slice_tsf_tensor(full_tensor, history_len, predict_len, n_samples=num_samples)
    else:
        data = generate_sine_sequence(num_samples, history_len + predict_len)

    data = data.to(device)
    x_history = data[:, :history_len, :]
    x_true = data[:, history_len:, :]

    # Output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "Data", "Predictions")
    os.makedirs(save_dir, exist_ok=True)

    # Sampling
    dt = -1.0 / num_diffusion_timesteps
    all_samples = []
    with torch.no_grad():
        for _ in tqdm(range(num_paths), desc="Sampling Trajectories", dynamic_ncols=True):
            x = torch.randn((num_samples, predict_len, input_dim), device=device)
            for i in reversed(range(num_diffusion_timesteps)):
                t_val = max(i / num_diffusion_timesteps, 1e-5)
                t = torch.full((num_samples, 1), t_val, device=device)
                score = model(x, x_history, t)
                drift = sde.f(x, t) - (sde.g(t) ** 2) * score
                z = torch.randn_like(x)
                x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z
            all_samples.append(x.cpu().numpy())

    all_samples = np.stack(all_samples, axis=0)  # [N_paths, B, T, D]

    # Create one stacked plot for all samples
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2.5 * num_samples), sharex=True)

    for sample_idx in range(num_samples):
        ax = axes[sample_idx]
        pred_samples = torch.tensor(all_samples[:, sample_idx, :, 0])  # [N_paths, T]
        mean = torch.mean(pred_samples, dim=0)
        lower = torch.quantile(pred_samples, 0.05, dim=0)
        upper = torch.quantile(pred_samples, 0.95, dim=0)
        true = x_true[sample_idx, :, 0].cpu()

        ax.plot(mean, label="Predictive Mean")
        ax.fill_between(range(predict_len), lower, upper, alpha=0.3, label="90% CI")
        ax.plot(true, label="True Future", linestyle="--")
        ax.set_title(f"Sample {sample_idx}")
        ax.legend()
        ax.grid(True)

        # Print metrics
        metrics = compute_all_metrics(mean, true, samples=pred_samples, alpha=0.9)
        metric_text = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        ax.text(1.01, 0.5, metric_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel("Time")
    plt.tight_layout()
    stacked_plot_path = os.path.join(save_dir, "stacked_predictions.png")
    plt.savefig(stacked_plot_path)
    plt.close()
    print(f"Saved stacked plot to: {stacked_plot_path}")

    print(f"\n✅ All distribution plots and metrics saved in: {save_dir}")

if __name__ == "__main__":
    predict_and_save_distribution(**CONFIG)
