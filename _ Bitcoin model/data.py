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
    t = torch.linspace(0, 2 * math.pi * sine_freq, total_seq_len)
    t = t.unsqueeze(0).expand(n_samples, -1)
    phase = 2 * math.pi * torch.rand(n_samples, 1)
    sine = sine_amplitude * torch.sin(t + phase)
    if seasonality:
        sine += 0.5 * sine_amplitude * torch.sin(2 * t + phase)

    trend_t = torch.arange(total_seq_len).float().unsqueeze(0).expand(n_samples, -1)
    if trend_type == "linear":
        trend = slope * trend_t
    elif trend_type == "quadratic":
        trend = slope * (trend_t ** 2)
    elif trend_type == "exp":
        trend = slope * torch.exp(trend_t / total_seq_len)
    else:
        raise ValueError(f"Unsupported trend_type: {trend_type}")

    if constant_variance:
        noise = noise_std * torch.randn_like(sine)
    else:
        if var_func is not None:
            variance_scale = var_func(trend_t / total_seq_len)
        else:
            variance_scale = 1 + trend_t / total_seq_len
        noise = noise_std * variance_scale * torch.randn_like(sine)

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

    signal = sine + trend + noise + jump_component
    signal = signal.unsqueeze(-1)

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
# 🔧 CSV LOADING FUNCTION
# ============================

def detect_normalization_factor(data, method="percentile", percentile=95):
    if method == "max":
        factor = np.max(np.abs(data))
    elif method == "percentile":
        factor = np.percentile(np.abs(data), percentile)
    elif method == "log":
        eps = 1e-8
        factor = np.exp(np.mean(np.log(np.abs(data) + eps)))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    print(f"✅ Detected normalization factor ({method}): {factor:.4f}")
    return factor

def load_csv_time_series(csv_path, history_len, predict_len, method="percentile"):
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)  # [T, N_samples]

    norm_factor = detect_normalization_factor(data, method=method)
    data = data / (norm_factor + 1e-8)

    data = torch.from_numpy(data).permute(1, 0).unsqueeze(-1)  # [N_samples, T, 1]
    return data, norm_factor


# ============================
# 📈 FORECAST EVALUATION METRICS
# ============================
def compute_mse(pred, true):
    return torch.mean((pred - true) ** 2).item()

def compute_mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()

def compute_rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()

def compute_smape(pred, true):
    denominator = (torch.abs(pred) + torch.abs(true)).clamp(min=1e-8)
    return 100.0 * torch.mean(2 * torch.abs(pred - true) / denominator).item()

def compute_nrmse(pred, true):
    range_val = (true.max() - true.min()).clamp(min=1e-8)
    return torch.sqrt(torch.mean((pred - true) ** 2)) / range_val

def compute_crps_ensemble(samples, true):
    N = samples.shape[0]
    true = true.unsqueeze(0).expand(N, -1)
    term1 = torch.mean(torch.abs(samples - true), dim=0)
    term2 = 0.5 * torch.mean(torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0)), dim=(0, 1))
    return torch.mean(term1 - term2).item()

def compute_interval_coverage(samples, true, alpha=0.9):
    lower = torch.quantile(samples, (1 - alpha) / 2, dim=0)
    upper = torch.quantile(samples, 1 - (1 - alpha) / 2, dim=0)
    within = (true >= lower) & (true <= upper)
    return torch.mean(within.float()).item()

def compute_all_metrics(pred, true, samples=None, alpha=0.9):
    metrics = {
        "MAE": compute_mae(pred, true),
        "MSE": compute_mse(pred, true),
        "RMSE": compute_rmse(pred, true),
        "SMAPE": compute_smape(pred, true),
        "NRMSE": compute_nrmse(pred, true).item(),
    }
    if samples is not None:
        metrics["CRPS"] = compute_crps_ensemble(samples, true)
        metrics[f"{int(alpha*100)}%_Coverage"] = compute_interval_coverage(samples, true, alpha)
    return metrics
