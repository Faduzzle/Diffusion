import torch

def generate_sine_sequence(n_samples, total_seq_len, input_dim=1):
    x = torch.linspace(0, 2 * torch.pi, total_seq_len)
    base = torch.sin(x).unsqueeze(0).repeat(n_samples, 1)
    base += 0.1 * torch.randn_like(base)
    return base.unsqueeze(-1)