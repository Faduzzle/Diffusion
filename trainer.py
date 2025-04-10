import torch
import numpy as np
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from sde import VPSDE
from data import generate_sine_sequence

sde = VPSDE()

def conditional_score_matching_loss(score_net, x_history, x_target, t, sde):
    mean, std = sde.p(x_target, t)
    z = torch.randn_like(x_target)
    x_t = mean + std * z
    score = score_net(x_t, x_history, t)
    loss = torch.mean(torch.sum((std * score + z) ** 2, dim=(1, 2)))
    return loss

def train_diffusion_model_conditional(data_x,
                                      score_net,
                                      optimizer,
                                      num_diffusion_timesteps,
                                      batch_size,
                                      num_epochs,
                                      history_len,
                                      predict_len,
                                      device,
                                      checkpoint_path=None,
                                      save_every=100,
                                      early_stopping_patience=20,
                                      visualize_every=20):

    score_net.train()
    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0

    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        losses = []
        perm = torch.randperm(len(data_x))

        for i in range(0, len(data_x), batch_size):
            end_idx = min(i + batch_size, len(data_x))
            idx = perm[i:end_idx]

            batch_seq = data_x[idx].to(device)
            x_history = batch_seq[:, :history_len, :]
            x_target = batch_seq[:, history_len:, :]
            current_batch_size = batch_seq.shape[0]

            assert x_target.shape[1] == predict_len, \
                f"Expected predict_len={predict_len}, got {x_target.shape[1]}"

            t = torch.rand((current_batch_size, 1), device=device)
            loss = conditional_score_matching_loss(score_net, x_history, x_target, t, sde)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0

            if checkpoint_path is not None:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                if checkpoint_dir and not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                torch.save({
                    'score_net_state_dict': score_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': avg_loss
                }, checkpoint_path)
                print(f"Best model checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Plot predictions every N epochs
        if (epoch + 1) % visualize_every == 0:
            score_net.eval()
            with torch.no_grad():
                num_samples = 5
                full_seq = generate_sine_sequence(num_samples, history_len + predict_len, input_dim=1).to(device)
                x_history = full_seq[:, :history_len, :]
                x_target_true = full_seq[:, history_len:, :]

                x = torch.randn((num_samples, predict_len, 1), device=device)
                dt = -1.0 / num_diffusion_timesteps

                for i in range(num_diffusion_timesteps - 1, -1, -1):
                    t = torch.full((num_samples, 1), i / num_diffusion_timesteps, device=device)
                    score = score_net(x, x_history, t)
                    drift = sde.f(x, t) - (sde.g(t) ** 2) * score
                    z = torch.randn_like(x)
                    x = x + drift * dt + sde.g(t) * ((-dt) ** 0.5) * z

                x_pred = x.cpu().squeeze(-1)
                x_history = x_history.cpu().squeeze(-1)
                x_target_true = x_target_true.cpu().squeeze(-1)

                for j in range(min(3, num_samples)):
                    plt.figure(figsize=(10, 4))
                    plt.plot(range(history_len), x_history[j], label='History', color='black')
                    plt.plot(range(history_len, history_len + predict_len), x_pred[j], label='Prediction', color='blue')
                    plt.plot(range(history_len, history_len + predict_len), x_target_true[j], label='Ground Truth', color='green', linestyle='dashed')
                    plt.title(f"Epoch {epoch + 1} - Sample {j}")
                    plt.xlabel("Time Step")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plot_dir = os.path.join(os.path.dirname(checkpoint_path), "sample_plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.savefig(os.path.join(plot_dir, f"epoch{epoch+1}_sample{j}.png"))
                    plt.close()
            score_net.train()

    # Always save final model
    if checkpoint_path is not None:
        torch.save({
            'score_net_state_dict': score_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': len(epoch_losses),
            'loss': epoch_losses[-1]
        }, checkpoint_path)
        print(f"Final model checkpoint saved to {checkpoint_path}")

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.splitext(checkpoint_path or "training_plot.png")[0] + "_loss_plot.png"
    plt.savefig(plot_path)
    print(f"Saved loss plot to {plot_path}")

    return epoch_losses
