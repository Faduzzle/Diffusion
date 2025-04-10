import torch
import numpy as np
from tqdm.auto import tqdm
import os
from sde import VPSDE

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
                                      save_every=100):

    score_net.train()
    epoch_losses = []
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        losses = []
        perm = torch.randperm(len(data_x))

        for i in range(0, len(data_x), batch_size):
            end_idx = min(i + batch_size, len(data_x))
            idx = perm[i:end_idx]

            batch_seq = data_x[idx].to(device)  # (B, seq_len, input_dim)
            x_history = batch_seq[:, :history_len, :]
            x_target = batch_seq[:, history_len:, :]
            current_batch_size = batch_seq.shape[0]

            t = torch.rand((current_batch_size, 1), device=device)

            loss = conditional_score_matching_loss(score_net, x_history, x_target, t, sde)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        if checkpoint_path is not None and ((epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs):
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            torch.save({
                'score_net_state_dict': score_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return epoch_losses
