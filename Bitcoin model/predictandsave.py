import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ScoreTransformerNet
from sde import VPSDE
from config import CONFIG
from data import load_csv_time_series, compute_all_metrics

def predict_and_save_inline(checkpoint_path, history_len=50, predict_len=20, input_dim=1,
                             num_windows=2, paths_per_window=2000, num_diffusion_timesteps=501):

    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ScoreTransformerNet(input_dim, history_len, predict_len).to(device)
    model.load_state_dict(ckpt["score_net_state_dict"])
    model.eval()

    sde = VPSDE()

    # Load test data from rolled CSV [num_windows, window_len, 1]
    data = load_csv_time_series(
        csv_path=CONFIG["test_csv_path"],
        history_len=history_len,
        predict_len=predict_len,
    ).to(device)  # shape [N_samples, window_len, 1]

    save_dir = os.path.join(os.path.dirname(__file__), "Data", "Predictions")
    os.makedirs(save_dir, exist_ok=True)

    # Sample column indices (i.e. which windows)
    col_indices = torch.randint(0, len(data), (num_windows,))
    pd.Series(col_indices.cpu().numpy(), name="column_idx").to_csv(os.path.join(save_dir, "selected_window_columns.csv"), index=False)
    print(f"✅ Saved selected column indices to: selected_window_columns.csv")

    selected_histories = data[col_indices, :history_len, :]
    selected_futures = data[col_indices, history_len:, :]

    expanded_histories = selected_histories.unsqueeze(1).expand(-1, paths_per_window, -1, -1)
    expanded_histories = expanded_histories.reshape(-1, history_len, input_dim)

    x = torch.randn((num_windows * paths_per_window, predict_len, input_dim), device=device)
    dt = -1.0 / num_diffusion_timesteps

    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1), desc="Sampling all windows"):
            t_val = max(i / num_diffusion_timesteps, 1e-5)
            t = torch.full((x.size(0), 1), t_val, device=device)
            score = model(x, expanded_histories, t)
            drift = sde.f(x, t) - (sde.g(t) ** 2) * score
            z = torch.randn_like(x)
            x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z

    x_pred_paths = x.cpu().view(num_windows, paths_per_window, predict_len)

    all_metrics = []
    fig, axes = plt.subplots(num_windows, 1, figsize=(14, 5 * num_windows))
    if num_windows == 1:
        axes = [axes]

    initial_price = 100.0

    for idx, (x_hist, x_true, ax) in enumerate(zip(selected_histories, selected_futures, axes)):
        sample_paths = x_pred_paths[idx]
        x_hist = x_hist.cpu().squeeze(-1)
        x_true = x_true.cpu().squeeze(-1)

        hist_price = initial_price * torch.exp(torch.cumsum(x_hist, dim=0))
        true_price = hist_price[-1] * torch.exp(torch.cumsum(x_true, dim=0))
        pred_prices = hist_price[-1] * torch.exp(torch.cumsum(sample_paths, dim=1))

        q10 = torch.quantile(pred_prices, 0.10, dim=0)
        q90 = torch.quantile(pred_prices, 0.90, dim=0)
        q35 = torch.quantile(pred_prices, 0.35, dim=0)
        q65 = torch.quantile(pred_prices, 0.65, dim=0)

        ax.plot(range(history_len), hist_price.numpy(), label="History", color="black")
        ax.plot(range(history_len, history_len + predict_len), true_price.numpy(), label="True Future", linestyle="--", color="green")
        ax.fill_between(range(history_len, history_len + predict_len), q10.numpy(), q90.numpy(), color="lightblue", alpha=0.3, label="10–90% Quantile Band")
        ax.fill_between(range(history_len, history_len + predict_len), q35.numpy(), q65.numpy(), color="blue", alpha=0.2, label="35–65% Quantile Band")

        ax.set_title(f"Sample {idx}")
        ax.legend()

        pred_mean = torch.mean(pred_prices, dim=0)
        metrics = compute_all_metrics(pred=pred_mean, true=true_price, samples=pred_prices, alpha=0.9)
        metrics["Sample"] = idx
        all_metrics.append(metrics)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "all_predictions.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"✅ Saved combined plot to: {fig_path}")

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
