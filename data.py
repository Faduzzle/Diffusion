import torch

def generate_sine_sequence(n_samples, hist_len, pred_len, input_dim=1, noise_std=0.1):
    total_seq_len = hist_len + pred_len
    x = torch.linspace(0, 2 * torch.pi, total_seq_len)
    shifts = 2 * torch.pi * torch.rand(n_samples, 1)
    x = x.unsqueeze(0) + shifts
    base = torch.sin(x).unsqueeze(-1)
    noise = noise_std * torch.randn_like(base)
    return base + noise
