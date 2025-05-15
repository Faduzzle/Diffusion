import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from model import ScoreTransformerNet
from sde import VPSDE
from config import CONFIG
from data import load_folder_as_tensor, SlidingWindowDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_crps(samples, truth):
    N = samples.shape[0]
    term1 = np.mean(np.abs(samples - truth), axis=0)
    term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]), axis=(0, 1))
    return np.mean(term1 - term2)

def compute_coverage(samples, truth, alpha=0.9):
    lower = np.quantile(samples, (1 - alpha) / 2, axis=0)
    upper = np.quantile(samples, 1 - (1 - alpha) / 2, axis=0)
    return np.mean((truth >= lower) & (truth <= upper))

def predict_and_save_inline(checkpoint_path, history_len=50, predict_len=20, input_dim=1,
                             num_windows=2, paths_per_window=2000, num_diffusion_timesteps=501):

    device = torch.device(CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️ Using device: {device}")

    guidance_w = CONFIG.get("classifier_free_guidance_weight", 2.0)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ScoreTransformerNet(input_dim, history_len, predict_len).to(device)
    model.load_state_dict(ckpt["ema_score_net_state_dict"] if "ema_score_net_state_dict" in ckpt else ckpt["score_net_state_dict"])
    model.eval()

    sde = VPSDE()

    # Load aligned test data
    test_tensor = load_folder_as_tensor(CONFIG["test_data_path"])
    if test_tensor is None or test_tensor.numel() == 0:
        raise ValueError("❌ Test tensor is empty. Check your test data formatting and overlap.")
    test_tensor = test_tensor.to(device)

    # Create test dataset and sample random windows
    test_dataset = SlidingWindowDataset(test_tensor, history_len, predict_len, mask_prob=0.0)
    indices = torch.randint(0, len(test_dataset), (num_windows,))
    subset = Subset(test_dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    selected_histories, selected_futures = [], []
    for hist, future in loader:
        selected_histories.append(hist.squeeze(0))
        selected_futures.append(future.squeeze(0))

    selected_histories = torch.stack(selected_histories)
    selected_futures = torch.stack(selected_futures)

    expanded_histories = selected_histories.unsqueeze(1).expand(-1, paths_per_window, -1, -1)
    expanded_histories = expanded_histories.reshape(-1, history_len, input_dim)

    x = torch.randn((num_windows * paths_per_window, predict_len, input_dim), device=device)
    dt = -1.0 / num_diffusion_timesteps

    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1), desc="Sampling all windows"):
            t_val = max(i / num_diffusion_timesteps, 1e-5)
            t = torch.full((x.size(0), 1), t_val, device=device)

            # Classifier-Free Guidance blending
            score_cond = model(x, expanded_histories, t, cond_drop_prob=0.0)
            score_uncond = model(x, torch.zeros_like(expanded_histories), t, cond_drop_prob=0.0)
            score = (1 + guidance_w) * score_cond - guidance_w * score_uncond

            drift = sde.f(x, t) - (sde.g(t) ** 2) * score
            z = torch.randn_like(x)
            x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z

    x_pred_paths = x.cpu().view(num_windows, paths_per_window, predict_len)

    # Save directory
    save_dir = os.path.join(r'C:\Users\thoma\Desktop\Diffusion\Bitcoin model\Predictions')
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(num_windows, 1, figsize=(14, 5 * num_windows))
    if num_windows == 1:
        axes = [axes]

    all_metrics = []

    for i, (hist, true, ax) in enumerate(zip(selected_histories, selected_futures, axes)):
        hist = hist.cpu().squeeze(-1)
        true = true.cpu().squeeze(-1)
        samples = x_pred_paths[i]

        hist_price = 100 * torch.exp(torch.cumsum(hist, dim=0))
        true_price = hist_price[-1] * torch.exp(torch.cumsum(true, dim=0))
        pred_prices = hist_price[-1] * torch.exp(torch.cumsum(samples, dim=1))

        q10 = torch.quantile(pred_prices, 0.10, dim=0)
        q90 = torch.quantile(pred_prices, 0.90, dim=0)
        q35 = torch.quantile(pred_prices, 0.35, dim=0)
        q65 = torch.quantile(pred_prices, 0.65, dim=0)

        ax.plot(range(history_len), hist_price.numpy(), label="History", color="black")
        ax.plot(range(history_len, history_len + predict_len), true_price.numpy(), label="True", linestyle="--", color="green")
        ax.fill_between(range(history_len, history_len + predict_len), q10.numpy(), q90.numpy(), color="lightblue", alpha=0.3, label="10–90% Quantile Band")
        ax.fill_between(range(history_len, history_len + predict_len), q35.numpy(), q65.numpy(), color="blue", alpha=0.2, label="35–65% Quantile Band")
        ax.set_title(f"Sample {i}")
        ax.legend()

        sample_dir = os.path.join(save_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, "log_return_paths.npy"), samples.numpy())
        price_paths = pred_prices.numpy()
        np.save(os.path.join(sample_dir, "price_paths.npy"), price_paths)

        pd.DataFrame({
            "q10": q10.numpy(),
            "q35": q35.numpy(),
            "q65": q65.numpy(),
            "q90": q90.numpy()
        }).to_csv(os.path.join(sample_dir, "quantile_bands.csv"), index=False)

        pred_mean = np.mean(price_paths, axis=0)
        mae = mean_absolute_error(true_price.numpy(), pred_mean)
        rmse = np.sqrt(mean_squared_error(true_price.numpy(), pred_mean))
        crps = compute_crps(price_paths, true_price.numpy())
        coverage = compute_coverage(price_paths, true_price.numpy(), alpha=0.9)

        pd.DataFrame({
            "pred_mean": pred_mean,
            "true_price": true_price.numpy()
        }).to_csv(os.path.join(sample_dir, "mean_vs_true.csv"), index=False)

        pd.Series({
            "MAE": mae,
            "RMSE": rmse,
            "CRPS": crps,
            "90%_Coverage": coverage
        }).to_csv(os.path.join(sample_dir, "metrics.csv"))

        all_metrics.append({
            "Sample": i,
            "MAE": mae,
            "RMSE": rmse,
            "CRPS": crps,
            "Coverage": coverage
        })

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "all_predictions.png")
    plt.savefig(fig_path)
    plt.close()

    full_metrics = pd.DataFrame(all_metrics)
    full_metrics.to_csv(os.path.join(save_dir, "all_metrics_summary.csv"), index=False)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    full_metrics[["MAE", "RMSE", "CRPS", "Coverage"]].hist(ax=axs.flatten(), bins=20, edgecolor='black')
    for ax in axs.flatten():
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metric_distributions.png"))
    plt.close()

    print(f"✅ Saved prediction plot and outputs to: {save_dir}")

if __name__ == "__main__":
    predict_and_save_inline(
        checkpoint_path=CONFIG["checkpoint_path"],
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"],
        input_dim=CONFIG.get("input_dim", 1),
        num_windows=5,
        paths_per_window=CONFIG["num_paths"],
        num_diffusion_timesteps=CONFIG["num_diffusion_timesteps"]
    )
