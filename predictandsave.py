import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from model import LatentDiffusionModel
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
    model = LatentDiffusionModel(input_dim, latent_dim=CONFIG.get("latent_dim", 32),
                                 model_dim=CONFIG.get("model_dim", 256)).to(device)
    model.load_state_dict(ckpt.get("ema_score_net_state_dict", ckpt["score_net_state_dict"]))
    model.eval()

    sde = VPSDE()

    # Load test data
    test_tensor = load_folder_as_tensor(CONFIG["test_data_path"]).to(device)
    test_dataset = SlidingWindowDataset(test_tensor, history_len, predict_len, mask_prob=0.0, dynamic_len=False)

    indices = torch.randint(0, len(test_dataset), (num_windows,))
    loader = DataLoader(Subset(test_dataset, indices.tolist()), batch_size=1, shuffle=False)

    selected_histories, selected_futures = [], []
    for hist, fut, *_ in loader:
        selected_histories.append(hist.squeeze(0))
        selected_futures.append(fut.squeeze(0))

    selected_histories = torch.stack(selected_histories).to(device)  # [B, H, D]
    selected_futures = torch.stack(selected_futures).to(device)      # [B, P, D]

    B, H, D = selected_histories.shape
    P = predict_len

    # Encode histories
    expanded_histories = selected_histories.unsqueeze(1).expand(-1, paths_per_window, -1, -1)
    expanded_histories = expanded_histories.reshape(-1, H, D)
    z_hist = model.encoder(expanded_histories)

    # Latent noise initialization
    z = torch.randn((num_windows * paths_per_window, P, model.score_net.output_proj.out_features), device=device)
    dt = -1.0 / num_diffusion_timesteps

    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1), desc="Sampling in latent space"):
            t_val = max(i / num_diffusion_timesteps, 1e-5)
            t = torch.full((z.size(0), 1), t_val, device=device)

            score_cond, _ = model.score_net(z, z_hist, t, cond_drop_prob=0.0)
            score_uncond, _ = model.score_net(z, torch.zeros_like(z_hist), t, cond_drop_prob=0.0)
            score = (1 + guidance_w) * score_cond - guidance_w * score_uncond

            drift = sde.f(z, t) - (sde.g(t) ** 2) * score
            z = z + drift * dt + sde.g(t) * ((-dt) ** 0.5) * torch.randn_like(z)

    # Decode latent predictions
    z_final = z.view(num_windows * paths_per_window, P, -1)
    x_pred = model.decode(z_final).cpu().view(num_windows, paths_per_window, P, D)

    # Save outputs
    save_dir = os.path.join(r'C:\Users\thoma\Desktop\Diffusion\Bitcoin model\Predictions')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_windows):
        hist = selected_histories[i].cpu().numpy()
        true = selected_futures[i].cpu().numpy()
        preds = x_pred[i].numpy()

        sample_dir = os.path.join(save_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)

        np.save(os.path.join(sample_dir, "pred_paths.npy"), preds)
        np.save(os.path.join(sample_dir, "history.npy"), hist)
        np.save(os.path.join(sample_dir, "truth.npy"), true)

        # Basic plot for first dimension
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(H), hist[:, 0], label="History", color="black")
        ax.plot(range(H, H + P), true[:, 0], label="True", linestyle="--", color="green")

        quantiles = {
            "q10": np.quantile(preds[:, :, 0], 0.10, axis=0),
            "q35": np.quantile(preds[:, :, 0], 0.35, axis=0),
            "q65": np.quantile(preds[:, :, 0], 0.65, axis=0),
            "q90": np.quantile(preds[:, :, 0], 0.90, axis=0),
        }

        ax.fill_between(range(H, H + P), quantiles["q10"], quantiles["q90"], color="lightblue", alpha=0.3, label="10–90%")
        ax.fill_between(range(H, H + P), quantiles["q35"], quantiles["q65"], color="blue", alpha=0.2, label="35–65%")
        ax.legend()
        ax.set_title(f"Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "forecast_plot.png"))
        plt.close()

        # Save metrics
        pred_mean = preds[:, :, 0].mean(axis=0)
        mae = mean_absolute_error(true[:, 0], pred_mean)
        rmse = np.sqrt(mean_squared_error(true[:, 0], pred_mean))
        crps = compute_crps(preds[:, :, 0], true[:, 0])
        coverage = compute_coverage(preds[:, :, 0], true[:, 0], alpha=0.9)

        pd.Series({
            "MAE": mae,
            "RMSE": rmse,
            "CRPS": crps,
            "90%_Coverage": coverage
        }).to_csv(os.path.join(sample_dir, "metrics.csv"))

    print(f"✅ Saved predictions to: {save_dir}")

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
