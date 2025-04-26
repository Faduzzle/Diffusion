import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ScoreTransformerNet
from sde import VPSDE
from config import CONFIG
from data import load_csv_time_series, compute_all_metrics

def unnormalize_predictions(predictions, norm_factor):
    return predictions * norm_factor

def predict_and_save_inline(checkpoint_path, history_len=50, predict_len=20, input_dim=1,
                             num_windows=2, paths_per_window=2000, num_diffusion_timesteps=501):

    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load model checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ScoreTransformerNet(input_dim, history_len, predict_len).to(device)
    model.load_state_dict(ckpt["score_net_state_dict"])
    model.eval()

    norm_factor = ckpt.get("norm_factor", 1.0)
    print(f"✅ Loaded normalization factor: {norm_factor:.6f}")

    sde = VPSDE()

    # Load real test set
    data, _ = load_csv_time_series(
        csv_path=CONFIG["test_csv_path"],
        history_len=history_len,
        predict_len=predict_len,
    )
    data = data.to(device)

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Predictions")
    os.makedirs(save_dir, exist_ok=True)

    indices = torch.randint(0, len(data), (num_windows,))
    selected_histories = data[indices, :history_len, :]
    selected_futures = data[indices, history_len:, :]

    # Expand all windows at once
    expanded_histories = selected_histories.unsqueeze(1).expand(-1, paths_per_window, -1, -1)
    expanded_histories = expanded_histories.reshape(num_windows * paths_per_window, history_len, input_dim)

    # Prepare initial noise
    x = torch.randn((num_windows * paths_per_window, predict_len, input_dim), device=device)
    dt = -1.0 / num_diffusion_timesteps

    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1),
                      desc="Sampling all windows", leave=False, dynamic_ncols=True):
            t_val = max(i / num_diffusion_timesteps, 1e-5)
            t = torch.full((num_windows * paths_per_window, 1), t_val, device=device)
            score = model(x, expanded_histories.to(device), t)
            drift = sde.f(x, t) - (sde.g(t) ** 2) * score
            z = torch.randn_like(x)
            x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z

    # Reshape outputs
    x_pred_paths = x.cpu().view(num_windows, paths_per_window, predict_len)  # [windows, paths, predict_len]

    all_metrics = []

    fig, axes = plt.subplots(num_windows, 1, figsize=(14, 5 * num_windows))
    if num_windows == 1:
        axes = [axes]

    for idx, (x_hist, x_true, ax) in enumerate(zip(selected_histories, selected_futures, axes)):
        # [paths_per_window, predict_len] for this sample
        sample_paths = x_pred_paths[idx]

        x_pred_mean = torch.mean(sample_paths, dim=0)
        ci_75_lower = torch.quantile(sample_paths, 0.125, dim=0)
        ci_75_upper = torch.quantile(sample_paths, 0.875, dim=0)
        ci_99_lower = torch.quantile(sample_paths, 0.005, dim=0)
        ci_99_upper = torch.quantile(sample_paths, 0.995, dim=0)

        x_hist_plot = x_hist.cpu().squeeze(-1)
        x_true_plot = x_true.cpu().squeeze(-1)

        # Unnormalize
        x_pred_mean = unnormalize_predictions(x_pred_mean, norm_factor)
        ci_75_lower = unnormalize_predictions(ci_75_lower, norm_factor)
        ci_75_upper = unnormalize_predictions(ci_75_upper, norm_factor)
        ci_99_lower = unnormalize_predictions(ci_99_lower, norm_factor)
        ci_99_upper = unnormalize_predictions(ci_99_upper, norm_factor)
        x_true_plot = unnormalize_predictions(x_true_plot, norm_factor)
        x_hist_plot = unnormalize_predictions(x_hist_plot, norm_factor)

        # Plot
        total_len = history_len + predict_len
        ax.plot(range(history_len), x_hist_plot.numpy(), label="History", color="black")
        ax.plot(range(history_len, total_len), x_true_plot.numpy(), label="True Future", linestyle="--", color="green")
        ax.plot(range(history_len, total_len), x_pred_mean.numpy(), label="Predicted Mean", color="blue")
        ax.fill_between(range(history_len, total_len), ci_99_lower.numpy(), ci_99_upper.numpy(), color="blue", alpha=0.2, label="99% CI")
        ax.fill_between(range(history_len, total_len), ci_75_lower.numpy(), ci_75_upper.numpy(), color="skyblue", alpha=0.4, label="75% CI")

        ax.set_title(f"Sample {idx}")
        ax.legend()

        # Compute metrics
        metrics = compute_all_metrics(
            pred=x_pred_mean,
            true=x_true_plot,
            samples=unnormalize_predictions(sample_paths, norm_factor),
            alpha=0.9,
        )
        metrics["Sample"] = idx
        all_metrics.append(metrics)

        # Table
        table_data = [[k, f"{v:.4f}"] for k, v in metrics.items() if k != "Sample"]
        table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "all_predictions.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Saved combined plot to: {fig_path}")

    # Save metrics CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(save_dir, "prediction_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✅ Saved metrics CSV to: {metrics_csv_path}")

if __name__ == "__main__":
    predict_and_save_inline(
        checkpoint_path=CONFIG["checkpoint_path"],
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"],
        input_dim=CONFIG["input_dim"],
        num_windows=5,
        paths_per_window=CONFIG["num_paths"],
        num_diffusion_timesteps=CONFIG["num_diffusion_timesteps"],
    )
