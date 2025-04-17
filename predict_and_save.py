import torch
import matplotlib.pyplot as plt
import pandas as pd
from sampler import sample

def predict_and_save(model, sde, data, hist_len, pred_len, save_path='predictions.csv'):
    with torch.no_grad():
        x_hist = data[:, :hist_len, :]
        x_true = data[:, hist_len:, :]
        x_pred = sample(model, sde, x_hist, pred_len)

        for i in range(min(5, data.size(0))):
            plt.figure(figsize=(10, 4))
            plt.plot(range(hist_len), x_hist[i].squeeze(), label="Historical")
            plt.plot(range(hist_len, hist_len + pred_len), x_true[i].squeeze(), label="True Future", linestyle='--')
            plt.plot(range(hist_len, hist_len + pred_len), x_pred[i].squeeze(), label="Predicted Future")
            plt.legend()
            plt.title(f"Sample {i}")
            plt.show()

        df = pd.DataFrame(x_pred.squeeze(-1).cpu().numpy())
        df.to_csv(save_path, index=False)
