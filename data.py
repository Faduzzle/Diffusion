import torch

def generate_sine_sequence(n_samples, total_seq_len, input_dim=1, noise_std=0.0):
    t = torch.linspace(0, 2 * torch.pi, total_seq_len)                      # [T]
    phase = 2 * torch.pi * torch.rand(n_samples, 1)                         # [B, 1]
    t_shifted = t.unsqueeze(0) + phase                                      # [B, T]
    base = torch.sin(t_shifted)                                             # [B, T]
    noise = noise_std * torch.randn(n_samples, total_seq_len)              # [B, T]
    signal = base + noise                                                   # [B, T]
    return signal.unsqueeze(-1)                                             # [B, T, 1]
