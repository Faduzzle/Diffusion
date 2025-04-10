import torch
import numpy as np
from tqdm import tqdm
from sde import VPSDE

def sample_conditional(score_net, x_history, predict_len, num_steps, device):
    sde = VPSDE()
    batch_size = x_history.size(0)
    input_dim = x_history.size(-1)
    x = torch.randn((batch_size, predict_len, input_dim), device=device)
    dt = -1.0 / num_steps

    with torch.no_grad():
        for i in tqdm(range(num_steps - 1, -1, -1), desc="Sampling"):
            t = torch.full((batch_size, 1), i / num_steps, device=device)
            score = score_net(x, x_history, t)
            drift = sde.f(x, t) - (sde.g(t) ** 2) * score
            z = torch.randn_like(x)
            x = x + drift * dt + sde.g(t) * np.sqrt(-dt) * z

    return x