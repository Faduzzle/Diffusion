import os
import time
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from config import CONFIG
from model import ScoreTransformerNet
from sde import VPSDE
from data import load_csv_time_series, generate_custom_series

def train(model, sde, data, history_len, predict_len, n_epochs=1000, batch_size=64, lr=1e-3,
          save_dir='checkpoints', checkpoint_freq=100, device="cuda", save_name="latest", norm_factor=1.0):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_history = data[:, :history_len, :].to(device)
    x_future = data[:, history_len:, :].to(device)

    dataset = TensorDataset(x_history, x_future)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    global_bar = tqdm(range(n_epochs), desc="Training", ncols=100)
    total_training_start = time.time()

    loss_history = []

    for epoch in global_bar:
        total_loss = 0.0
        model.train()

        for hist, future in loader:
            t = torch.rand(hist.size(0), device=device) * 0.998 + 0.001
            t_expand = t.unsqueeze(1).expand(-1, future.size(1))

            mean, std = sde.p(future, t_expand)
            noise = torch.randn_like(std)
            x_t = mean + std * noise

            score_pred = model(x_t, hist, t.unsqueeze(1))
            loss = torch.mean(torch.pow(std * score_pred + noise, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        global_bar.set_postfix(loss=avg_loss)

        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == n_epochs:
            epoch_path = os.path.join(save_dir, f"model_epoch_{epoch+1:04d}.pt")
            latest_path = os.path.join(save_dir, f"{save_name}.pth")

            torch.save({
                'score_net_state_dict': model.state_dict(),
                'norm_factor': norm_factor,
            }, epoch_path)

            torch.save({
                'score_net_state_dict': model.state_dict(),
                'norm_factor': norm_factor,
            }, latest_path)

    total_time = time.time() - total_training_start
    print(f"\n✅ Training complete in {total_time:.2f} seconds ({total_time/60:.2f} min)")

    # Save training loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    loss_plot_path = os.path.join(save_dir, f"{save_name}_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    print(f"📈 Saved training loss curve to: {loss_plot_path}")

def train_model_from_config():
    device = CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    if CONFIG.get("use_csv"):
        data, norm_factor = load_csv_time_series(
            csv_path=CONFIG["train_csv_path"],
            history_len=CONFIG["history_len"],
            predict_len=CONFIG["predict_len"],
        )
    elif CONFIG.get("use_synthetic"):
        data = generate_custom_series(
            n_samples=CONFIG["n_samples"],
            total_seq_len=CONFIG["history_len"] + CONFIG["predict_len"],
            input_dim=CONFIG["input_dim"],
            sine_amplitude=CONFIG.get("sine_amplitude", 1.0),
            sine_freq=CONFIG.get("sine_freq", 1.0),
            slope=CONFIG.get("slope", 0.0),
            trend_type=CONFIG.get("trend_type", "linear"),
            noise_std=CONFIG.get("noise_std", 0.1),
            constant_variance=CONFIG.get("constant_variance", True),
            jumps=CONFIG.get("jumps", False),
            seasonality=CONFIG.get("seasonality", False),
        )
        norm_factor = 1.0
    else:
        raise ValueError("Specify either 'use_csv' or 'use_synthetic' in CONFIG.")

    model = ScoreTransformerNet(
        input_dim=CONFIG["input_dim"],
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"],
        model_dim=CONFIG.get("model_dim", 256),
    ).to(device)

    sde = VPSDE(bmin=0.1, bmax=20.0)

    train(
        model, sde, data,
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"],
        n_epochs=CONFIG["n_epochs"],
        batch_size=CONFIG["batch_size"],
        lr=CONFIG["lr"],
        save_dir=CONFIG["checkpoint_dir"],
        checkpoint_freq=CONFIG["checkpoint_freq"],
        save_name=CONFIG["save_name"],
        device=device,
        norm_factor=norm_factor
    )

if __name__ == "__main__":
    train_model_from_config()
