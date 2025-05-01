import torch

def load_processed_dataset(path):
    """
    Loads prerolled, pre-normalized training dataset.

    Args:
        path (str): Path to saved .pt file

    Returns:
        windows (Tensor): [num_samples, total_seq_len, n_features]
        norm_params (dict): parameters for de-normalization later
        normalization_mode (str): how data was normalized
        timestamp (str): when dataset was created
    """
    data = torch.load(path)

    windows = data["windows"]           # [num_samples, total_seq_len, n_features]
    norm_params = data["norm_params"]    # dict: keys depend on mode (max, standard, etc.)
    normalization_mode = data.get("normalization_mode", "unknown")
    timestamp = data.get("timestamp", "Unknown")

    print(f"✅ Loaded {windows.shape[0]} samples, each with {windows.shape[1]} timesteps and {windows.shape[2]} features.")
    print(f"📅 Dataset created on: {timestamp} | Normalization used: {normalization_mode}")

    return windows, norm_params, normalization_mode, timestamp


def apply_normalization(windows, norm_params, mode="max"):
    """
    Applies the selected normalization method.

    Args:
        windows (Tensor): [num_samples, total_seq_len, n_features]
        norm_params (dict): normalization parameters
        mode (str): one of ['max', 'standard', 'percentile', 'exp', 'none']

    Returns:
        normalized_windows (Tensor): normalized data
    """
    if mode == "max":
        # Divide each feature by its maximum absolute value
        norm_values = norm_params["values"]  # [n_features]
        normalized = windows / (norm_values + 1e-8)

    elif mode == "standard":
        # Standardize each feature to zero mean, unit variance
        mean = norm_params["mean"]
        std = norm_params["std"]
        normalized = (windows - mean) / (std + 1e-8)

    elif mode == "percentile":
        # Divide by 99th (or chosen) percentile
        norm_values = norm_params["percentile"]
        normalized = windows / (norm_values + 1e-8)

    elif mode == "exp":
        # Apply log transform
        normalized = torch.log(windows + 1e-8)

    elif mode == "none":
        # No normalization applied
        normalized = windows

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    print(f"🔵 Applied '{mode}' normalization to windows.")
    return normalized
