import torch
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F

# ============================
# 🔧 CUSTOM TIME SERIES GENERATOR
# ============================
def generate_custom_series(n_samples, total_seq_len, input_dim=1,
                            sine_amplitude=1.0, sine_freq=1.0,
                            slope=0.0, trend_type="linear",
                            noise_std=0.1, constant_variance=True, var_func=None,
                            jumps=False, spike_prob=0.02, jump_scale=3.0, jump_tau=5,
                            seasonality=False,
                            return_components=False):
    
    t = torch.linspace(0, 2 * math.pi * sine_freq, total_seq_len)  # [T]
    t = t.unsqueeze(0).expand(n_samples, -1)  # [B, T]

    # Sine with random phase shift
    phase = 2 * math.pi * torch.rand(n_samples, 1)
    sine = sine_amplitude * torch.sin(t + phase)

    if seasonality:
        sine += 0.5 * sine_amplitude * torch.sin(2 * t + phase)

    # Trend
    trend_t = torch.arange(total_seq_len).float().unsqueeze(0).expand(n_samples, -1)
    if trend_type == "linear":
        trend = slope * trend_t
    elif trend_type == "quadratic":
        trend = slope * (trend_t ** 2)
    elif trend_type == "exp":
        trend = slope * torch.exp(trend_t / total_seq_len)
    else:
        raise ValueError(f"Unsupported trend_type: {trend_type}")

    # Noise
    if constant_variance:
        noise = noise_std * torch.randn_like(sine)
    else:
        if var_func is not None:
            variance_scale = var_func(trend_t / total_seq_len)
        else:
            variance_scale = 1 + trend_t / total_seq_len
        noise = noise_std * variance_scale * torch.randn_like(sine)

    # Jumps with exponential decay (causal)
    if jumps:
        spike_mask = (torch.rand(n_samples, total_seq_len) < spike_prob).float()
        spike_magnitude = torch.randn(n_samples, total_seq_len) * noise_std * jump_scale

        kernel_len = min(20, total_seq_len)
        decay_kernel = torch.exp(-torch.arange(0, kernel_len).float() / jump_tau)
        decay_kernel = decay_kernel.flip(0).to(sine.device).view(1, 1, -1)

        jump_component = torch.zeros_like(sine)
        for b in range(n_samples):
            spike_signal = (spike_mask[b] * spike_magnitude[b]).view(1, 1, -1)
            padded = F.pad(spike_signal, (0, kernel_len - 1))
            convolved = F.conv1d(padded, decay_kernel)
            jump_component[b] = convolved.squeeze(0).squeeze(0)[:total_seq_len]
    else:
        jump_component = torch.zeros_like(sine)

    # Final signal
    signal = sine + trend + noise + jump_component
    signal = signal.unsqueeze(-1)  # [B, T, 1]

    if return_components:
        return {
            "signal": signal,
            "sine": sine.unsqueeze(-1),
            "trend": trend.unsqueeze(-1),
            "noise": noise.unsqueeze(-1),
            "jumps": jump_component.unsqueeze(-1),
        }

    return signal



# ============================
# 🔧 DEFAULT DATA CONFIGURATION
# ============================

default_data_config = {
    "n_samples": 1000,             # Number of sequences to generate
    "total_seq_len": 100,          # Total time steps (history + future)
    "input_dim": 1,                # Input feature dimension (usually 1 for univariate)

    # === Sine Wave Parameters ===
    "sine_amplitude": 1.0,         # Amplitude of the sine wave
    "sine_freq": 1.0,              # Frequency of the sine wave (in cycles over the sequence)

    # === Trend Settings ===
    "slope": 0.01,                 # Slope of the trend component
    "trend_type": "linear",        # Choose from: "linear", "quadratic", "exp"

    # === Noise Parameters ===
    "noise_std": 0.1,              # Standard deviation of Gaussian noise
    "constant_variance": True,     # If False, variance changes over time via var_func

    # === Optional Variance Scaling Function ===
    # If constant_variance=False, use this to control noise shape over time.
    # t ∈ [0, 1] represents normalized time. Output scales the noise std.
    # Examples:
    #   lambda t: 1 + t            → Linearly increasing variance
    #   lambda t: 2 - t            → Linearly decreasing variance
    #   lambda t: 1 + t**2         → Quadratic growth
    #   lambda t: torch.exp(t)     → Exponential growth
    #   lambda t: 1 + (t - 0.5)**2 → U-shaped: high at start/end, low in middle
    #   lambda t: 1 + 0.5 * torch.sin(2 * math.pi * t) → Cyclical variance
    "var_func": None,

    # === Advanced Signal Features ===
    "seasonality": False,          # If True, adds higher frequency seasonal component
    "jumps": False,                # If True, randomly inserts large spikes
    "jump_scale":3.0,              # Scale of the jumps (e.g., 3.0 means 3x noise_std)
    "jump_tau":5,                  # Decay time constant for jumps (larger = slower decay)

    # === Output Option ===
    "return_components": False     # If True, returns a dict of signal + parts (for visualization/debugging)
    
    }

# ============================
# 🔧 CSV LOADING FUNCTION
# ============================


def load_csv_time_series(
    csv_path,
    history_len,
    predict_len,
    normalize=True,
    fill_method="ffill",
    n_samples=1000,
    stride=1,
    return_raw=False
):
    """
    Load time series from CSV and return sliced tensor samples.

    Args:
        csv_path (str): Path to time series CSV (rows = time, cols = variables).
        history_len (int): Number of steps for historical input.
        predict_len (int): Number of steps to forecast.
        normalize (bool): Whether to apply z-score normalization.
        fill_method (str): How to fill missing values: "ffill", "bfill", or None.
        n_samples (int): Number of training samples to draw.
        stride (int): Sampling stride.
        return_raw (bool): If True, returns full cleaned data as well.

    Returns:
        Tensor: [n_samples, history_len + predict_len, D] float tensor
        Optional: Raw cleaned DataFrame
    """
    df = pd.read_csv(csv_path, index_col=0 if "date" in csv_path.lower() else None)
    
    # Fill missing values
    if fill_method:
        df = df.fillna(method=fill_method)

    # Normalize per column
    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-8)

    data = df.to_numpy(dtype=np.float32)
    T, D = data.shape
    window_size = history_len + predict_len

    if T < window_size:
        raise ValueError("Time series too short for specified window size.")

    max_start = T - window_size
    starts = np.random.choice(np.arange(0, max_start, stride), size=n_samples, replace=True)

    sequences = np.stack([data[s:s+window_size] for s in starts])  # [B, T, D]
    tensor = torch.tensor(sequences)  # [B, T, D]

    return (tensor, df) if return_raw else tensor
