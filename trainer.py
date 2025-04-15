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
                                      ):

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
