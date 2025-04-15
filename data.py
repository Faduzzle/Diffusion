import torch

def generate_sine_sequence(n_samples, total_seq_len, input_dim=1, noise_std=0.1):
    t = torch.linspace(0, 2 * torch.pi, total_seq_len)  # (seq_len,)
    phase = 2 * torch.pi * torch.rand(n_samples).unsqueeze(1)  # (n_samples, 1)
    t_shifted = t.unsqueeze(0) + phase  # (n_samples, seq_len)

    base = torch.sin(t_shifted) + noise_std * torch.randn(n_samples, total_seq_len)
    return base.unsqueeze(-1)  # (n_samples, seq_len, 1)
