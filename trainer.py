import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def train(model, sde, data, hist_len, pred_len, n_epochs=1000, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_hist = data[:, :hist_len, :]
    x_future = data[:, hist_len:, :]

    dataset = TensorDataset(x_hist, x_future)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        for hist, future in loader:
            t = torch.rand(hist.size(0), 1).to(hist.device)
            mean, std = sde.p(future, t)
            noise = torch.randn_like(std)
            x_t = mean + std * noise

            score_pred = model(x_t, t, hist)
            target = -noise / std.squeeze(-1)
            loss = F.mse_loss(score_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
