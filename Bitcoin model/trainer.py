import os
import time
import copy
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CONFIG
from model import ScoreTransformerNet
from sde import VPSDE
from data import SlidingWindowDataset, load_folder_as_tensor

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def train(model, sde, dataset, history_len, predict_len, n_epochs=1000, batch_size=64, lr=1e-3,
          save_dir='checkpoints', checkpoint_freq=100, device="cuda", save_name="latest", ema_decay=0.999):

    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = []
    global_bar = tqdm(range(n_epochs), desc="Training", ncols=100)
    start_time = time.time()

    cond_drop_prob = CONFIG.get("cond_drop_prob", 0.1)

    for epoch in global_bar:
        model.train()
        total_loss = 0.0

        for hist, future in loader:
            hist, future = hist.to(device), future.to(device)
            t = torch.rand(hist.size(0), device=device) * 0.998 + 0.001
            t_expand = t.unsqueeze(1).expand(-1, future.size(1))

            mean, std = sde.p(future, t_expand)
            noise = torch.randn_like(std)
            x_t = mean + std * noise

            score_pred = model(x_t, hist, t.unsqueeze(1), cond_drop_prob=cond_drop_prob)
            loss = torch.mean((std * score_pred + noise) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, ema_decay)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        global_bar.set_postfix(loss=avg_loss)

        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == n_epochs:
            torch.save({
                "score_net_state_dict": model.state_dict(),
                "ema_score_net_state_dict": ema_model.state_dict()
            }, os.path.join(save_dir, f"{save_name}.pth"))

    duration = time.time() - start_time
    print(f"✅ Training completed in {duration:.2f} sec ({duration/60:.2f} min)")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{save_name}_loss_curve.png"))
    plt.close()

def train_model_from_config():
    device = CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    history_len = CONFIG["history_len"]
    predict_len = CONFIG["predict_len"]

    train_tensor = load_folder_as_tensor(CONFIG["train_data_path"])

    print("🧪 Loaded train_tensor shape:", train_tensor.shape)

    dataset = SlidingWindowDataset(
        data_tensor=train_tensor,
        history_len=history_len,
        predict_len=predict_len,
        mask_prob=CONFIG.get("mask_prob", 0.1)
    )

    model = ScoreTransformerNet(
        input_dim=train_tensor.shape[-1],
        history_len=history_len,
        predict_len=predict_len,
        model_dim=CONFIG.get("model_dim", 256)
    ).to(device)

    sde = VPSDE()
    train(model, sde, dataset,
          history_len=history_len,
          predict_len=predict_len,
          n_epochs=CONFIG["n_epochs"],
          batch_size=CONFIG["batch_size"],
          lr=CONFIG["lr"],
          save_dir=CONFIG["checkpoint_dir"],
          checkpoint_freq=CONFIG["checkpoint_freq"],
          save_name=CONFIG["save_name"],
          device=device)

if __name__ == "__main__":
    train_model_from_config()
