import torch

def sample(model, sde, hist, pred_len, steps=100):
    model.eval()
    device = next(model.parameters()).device
    batch_size = hist.size(0)
    x = torch.randn(batch_size, pred_len, 1).to(device)
    ts = torch.linspace(1., 1e-3, steps).to(device)
    dt = -1. / steps

    for t in ts:
        t_batch = torch.full((batch_size, 1), t).to(device)
        score = model(x, t_batch, hist)
        drift = sde.f(x, t_batch)
        diffusion = sde.g(t_batch).view(-1, 1, 1)
        x = x + (drift - diffusion**2 * score.unsqueeze(-1)) * dt + diffusion * torch.sqrt(-dt) * torch.randn_like(x)

    return x.detach()
