import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import ConditionalScoreNet
from sde import VPSDE
from sampler import sample_conditional
from data import generate_sine_sequence



# === Settings ===
history_len = 150
predict_len = 50
seq_len = history_len + predict_len
num_samples = 50
checkpoint_path = "checkpoints\score_model.pth"
output_dir = "Data/Predictions"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions.csv")
output_path_T = os.path.join(output_dir, "predictions_transposed.csv")

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
model = ConditionalScoreNet(input_dim=1, history_len=history_len, predict_len=predict_len).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["score_net_state_dict"])
model.eval()

# === Generate input data ===
history = generate_sine_sequence(num_samples, history_len, input_dim=1).to(device)

# === Sample predictions ===
with torch.no_grad():
    predicted = sample_conditional(model, history, predict_len=predict_len, num_steps=500, device=device)

# === Combine and label columns ===
full_sequence = torch.cat([history, predicted], dim=1).squeeze(-1).cpu().numpy()
columns = [f"h_{i}" for i in range(history_len)] + [f"p_{i}" for i in range(predict_len)]
df = pd.DataFrame(full_sequence, columns=columns)
df.to_csv(output_path, index=False)
print(f"Saved predictions to {output_path}")

# === Save transposed version ===
df.T.to_csv(output_path_T, header=True)
print(f"Saved transposed predictions to {output_path_T}")

# === Plot a few examples ===
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

for i in range(5):
    plt.figure(figsize=(10, 4))
    plt.plot(range(history_len), full_sequence[i, :history_len], label="History")
    plt.plot(range(history_len, seq_len), full_sequence[i, history_len:], label="Prediction")
    plt.title(f"Sample {i}")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"sample_{i}.png"))
    plt.close()

print(f"Saved example plots to {plot_dir}")
