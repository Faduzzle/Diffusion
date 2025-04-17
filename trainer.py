import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

def train(model, sde, data, history_len, predict_len, n_epochs=1000, batch_size=64, lr=1e-3,
          save_dir='checkpoints', checkpoint_freq=100, use_score_matching=True):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_history = data[:, :history_len, :]   # [B, history_len, 1]
    x_future = data[:, history_len:, :]    # [B, predict_len, 1]

    dataset = TensorDataset(x_history, x_future)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    global_bar = tqdm(range(n_epochs), desc="Training", ncols=100)
    total_training_start = time.time()

    for epoch in global_bar:
        total_loss = 0.0
        model.train()

        for hist, future in loader:
            hist = hist.to(next(model.parameters()).device)
            future = future.to(next(model.parameters()).device)

            # Sample diffusion time and expand to future length
            t = torch.rand(hist.size(0)).to(hist.device) * 0.998 + 0.001         # [B]
            t_expand = t.unsqueeze(1).expand(-1, future.size(1))                 # [B, pred_len]

            # Get noised future
            mean, std = sde.p(future, t_expand)                                  # [B, pred_len, 1]
            noise = torch.randn_like(std)
            x_t = mean + std * noise                                             # [B, pred_len, 1]

            # Forward through model
            score_pred = model(x_t, hist, t.unsqueeze(1))                        # [B, pred_len, 1]

            # Toggle: score matching vs. denoising
            if use_score_matching:
                target = -noise / std                                            # score = -ε / σ
                target = torch.clamp(target, -5, 5)                              # stabilize exploding scores
            else:
                target = future                                                  # denoising target

            # Sanity check
            assert score_pred.shape == target.shape, \
                f"Shape mismatch: pred={score_pred.shape}, target={target.shape}"

            # Loss
            loss = F.mse_loss(score_pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)    # clip to avoid exploding gradients
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        global_bar.set_postfix(loss=avg_loss)

        # Save checkpoints
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == n_epochs:
            epoch_path = os.path.join(save_dir, f"model_epoch_{epoch+1:04d}.pt")
            latest_path = os.path.join(save_dir, "latest.pth")
            torch.save({'score_net_state_dict': model.state_dict()}, epoch_path)
            torch.save({'score_net_state_dict': model.state_dict()}, latest_path)

    total_time = time.time() - total_training_start
    print(f"\n✅ Training complete in {total_time:.2f} seconds ({total_time/60:.2f} min)")
