# ========= IMPORTS =========
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CONFIG
from data import load_processed_dataset
from model import ScoreTransformerNet
from sde import VPSDE

# ========= SMALL FIX =========
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ========= METRICS =========

def compute_metrics(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    mse = torch.mean((y_true - y_pred) ** 2).item()
    mape = (torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100).item()
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "MAPE (%)": mape
    }

# ========= SAMPLING FUNCTION =========

@torch.no_grad()
def sample_predictions(model, sde, x_history, t_steps, n_samples):
    model.eval()
    device = x_history.device
    predict_len = CONFIG["predict_len"]
    input_dim = x_history.size(-1)

    x = torch.randn(n_samples, predict_len, input_dim, device=device)

    for t in t_steps:
        t_batch = torch.ones(n_samples, 1, device=device) * t

        alpha = sde.alpha(t_batch)
        sigma = torch.sqrt(1.0 - alpha ** 2)

        score = model(x, x_history.expand(n_samples, -1, -1), t_batch)

        noise = torch.randn_like(x) if t > 0 else 0.0
        x = (x + sigma * score) / alpha + sigma * noise

    return x

# ========= PLOTTING FUNCTION =========

def plot_feature(time_idx, true_future, pred_samples, feature_idx, window_idx, save_dir):
    """
    Plot a single feature with confidence intervals.
    """
    pred_mean = pred_samples.mean(dim=0)[:, feature_idx]
    pred_std = pred_samples.std(dim=0)[:, feature_idx]

    ci_99 = 2.576 * pred_std
    ci_75 = 1.150 * pred_std

    fig, ax = plt.subplots(figsize=(8, 5))
    time_idx_future = time_idx[-CONFIG["predict_len"]:]

    ax.plot(time_idx_future, true_future[:, feature_idx].cpu(), label="True", linestyle='-', marker='o')
    ax.plot(time_idx_future, pred_mean.cpu(), label="Pred Mean", linestyle='--')

    ax.fill_between(time_idx_future,
                    (pred_mean - ci_75).cpu(),
                    (pred_mean + ci_75).cpu(),
                    alpha=0.3, label="75% CI")

    ax.fill_between(time_idx_future,
                    (pred_mean - ci_99).cpu(),
                    (pred_mean + ci_99).cpu(),
                    alpha=0.1, label="99% CI")

    ax.set_title(f"Window {window_idx} - Feature {feature_idx}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Value")
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"window{window_idx}_feature{feature_idx}.png"))
    plt.close()

# ========= MAIN SCRIPT =========

if __name__ == "__main__":
    # Load config
    checkpoint_path = CONFIG["checkpoint_path"]
    predictions_per_window = CONFIG["predictions_per_window"]
    save_name = CONFIG["save_name"]

    # Setup output paths
    output_dir = "./Model_Output"
    os.makedirs(output_dir, exist_ok=True)
    result_csv_path = os.path.join(output_dir, f"{save_name}_results.csv")
    plot_dir = os.path.join(output_dir, f"{save_name}_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Load dataset
    windows, norm_params, normalization_mode, timestamp = load_processed_dataset(CONFIG["processed_dataset_path"])
    test_windows = windows

    # Load model
    input_dim = test_windows.shape[-1]
    model = ScoreTransformerNet(
        input_dim=input_dim,
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"],
        model_dim=CONFIG["model_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
    ).to(CONFIG["device"])

    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load diffusion
    sde = VPSDE()

    # Sample 3 random windows
    chosen_indices = torch.randint(0, test_windows.shape[0], (3,))
    print(f"📈 Sampling windows: {chosen_indices.tolist()}")

    # Prepare to store all metrics
    all_metrics = []

    for idx, window_idx in enumerate(chosen_indices):
        window = test_windows[window_idx]
        x_history = window[:CONFIG["history_len"], :].unsqueeze(0).to(CONFIG["device"])
        true_future = window[CONFIG["history_len"]:, :]

        t_steps = torch.linspace(1.0, 0.0, steps=CONFIG["num_diffusion_timesteps"], device=CONFIG["device"])
        pred_samples = sample_predictions(model, sde, x_history, t_steps, predictions_per_window)

        time_idx = torch.arange(window.shape[0])

        # For each feature separately
        for feature_idx in range(input_dim):
            feature_save_dir = os.path.join(plot_dir, f"window{window_idx}_feature{feature_idx}")
            plot_feature(time_idx, true_future, pred_samples, feature_idx, window_idx, feature_save_dir)

            # Compute metrics for this feature
            pred_mean = pred_samples.mean(dim=0)[:, feature_idx]
            metrics = compute_metrics(true_future[:, feature_idx].to(pred_mean.device), pred_mean)

            # Add metadata
            metrics["Window_Index"] = int(window_idx.item())
            metrics["Feature_Index"] = int(feature_idx)
            all_metrics.append(metrics)

            # Print Metrics
            print(f"\n📋 Metrics for Window {window_idx} - Feature {feature_idx}:")
            for k, v in metrics.items():
                if k not in ("Window_Index", "Feature_Index"):
                    print(f"{k}: {v:.6f}")
            print("\n" + "="*40 + "\n")

    # Save/Append Results CSV
    new_results = pd.DataFrame(all_metrics)
    if os.path.exists(result_csv_path):
        existing_results = pd.read_csv(result_csv_path)
        final_results = pd.concat([existing_results, new_results], ignore_index=True)
    else:
        final_results = new_results

    final_results.to_csv(result_csv_path, index=False)
    print(f"✅ Saved evaluation results to {result_csv_path}")
