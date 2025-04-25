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
from config import CONFIG

def predict_and_save_distribution():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ScoreTransformerNet(
        input_dim=CONFIG["input_dim"],
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"]
    ).to(device)

    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["score_net_state_dict"])
    model.eval()

    sde = VPSDE()

    if CONFIG["use_tsf"]:
        full_tensor = load_tsf(CONFIG["tsf_path"])
        data = slice_tsf_tensor(full_tensor, CONFIG["history_len"], CONFIG["predict_len"], n_samples=CONFIG["n_samples"])
    else:
        data = generate_sine_sequence(CONFIG["n_samples"], CONFIG["history_len"] + CONFIG["predict_len"])

    data = data.to(device)
    x_history = data[:, :CONFIG["history_len"], :]
    x_true = data[:, CONFIG["history_len"]:, :]

    dt = -1.0 / CONFIG["num_diffusion_timesteps"]
    all_samples = []

    with torch.no_grad():
        for _ in tqdm(range(CONFIG["num_paths"]), desc="Sampling Trajectories", dynamic_ncols=True):
            x = torch.randn((CONFIG["n_samples"], CONFIG["predict_len"], CONFIG["input_dim"]), device=device)
            for i in reversed(range(CONFIG["num_diffusion_timesteps"])):
                t_val = max(i / CONFIG["num_diffusion_timesteps"], 1e-5)
                t = torch.full((CONFIG["n_samples"], 1), t_val, device=device)
                score = model(x, x_history, t)
                drift = sde.f(x, t) - (sde.g(t) ** 2) * score
                z = torch.randn_like(x)
                x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z
            all_samples.append(x.cpu().numpy())

    all_samples = np.stack(all_samples, axis=0)

    save_dir = os.path.join("Data", "Predictions")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(CONFIG["n_samples"], 1, figsize=(10, 2.5 * CONFIG["n_samples"]), sharex=True)

    for sample_idx in range(CONFIG["n_samples"]):
        ax = axes[sample_idx]
        pred_samples = torch.tensor(all_samples[:, sample_idx, :, 0])
        mean = torch.mean(pred_samples, dim=0)
        lower = torch.quantile(pred_samples, 0.05, dim=0)
        upper = torch.quantile(pred_samples, 0.95, dim=0)
        true = x_true[sample_idx, :, 0].cpu()

        ax.plot(mean, label="Predictive Mean")
        ax.fill_between(range(CONFIG["predict_len"]), lower, upper, alpha=0.3, label="90% CI")
        ax.plot(true, label="True Future", linestyle="--")
        ax.set_title(f"Sample {sample_idx}")
        ax.legend()
        ax.grid(True)

        metrics = compute_all_metrics(mean, true, samples=pred_samples, alpha=0.9)
        metric_text = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        ax.text(1.01, 0.5, metric_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel("Time")
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "stacked_predictions.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved stacked plot to: {plot_path}")

if __name__ == "__main__":
    predict_and_save_distribution()
