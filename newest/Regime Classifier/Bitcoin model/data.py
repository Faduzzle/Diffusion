import os
import pandas as pd
import numpy as np
import torch
from functools import reduce
from wavelet_utils import modwt_decompose
from torch.utils.data import Dataset

def load_folder_as_tensor(folder_path):
    series, date_sets = [], []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            required_cols = {"Year", "Month", "Day", "Value"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"❌ File {file} is missing required columns: {required_cols - set(df.columns)}")
            df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
            df = df.sort_values("Date")[["Date", "Value"]].rename(columns={"Value": file})
            series.append(df)
            date_sets.append(set(df["Date"]))

    if not series:
        raise ValueError(f"❌ No valid CSV files found in {folder_path}")

    shared_dates = sorted(reduce(set.intersection, date_sets))
    merged = pd.DataFrame({"Date": shared_dates})

    for s in series:
        merged = merged.merge(s, on="Date", how="left")

    print(f"🔎 Before dropna: {merged.shape}")
    merged = merged.dropna()
    print(f"🔎 After dropna: {merged.shape}")

    data_array = merged.drop(columns=["Date"]).values.astype(np.float32)
    return torch.from_numpy(data_array)


class SlidingWindowDataset(Dataset):
    def __init__(self, data_tensor, history_len, predict_len, mask_prob=0.01):
        self.data = data_tensor
        self.history_len = history_len
        self.predict_len = predict_len
        self.mask_prob = mask_prob
        self.total_len = history_len + predict_len
        
        # Ensure data is long enough
        if len(self.data) < self.total_len:
            raise ValueError("Data length must be >= history_len + predict_len")
            
        # Calculate all possible start indices
        self.max_start = len(self.data) - self.total_len + 1
            
    def __len__(self):
        return self.max_start  # Return total number of possible windows
        
    def __getitem__(self, idx):
        # Use the index directly - random sampling is handled by DataLoader
        start_idx = idx
        end_idx = start_idx + self.total_len
        window = self.data[start_idx:end_idx]
        
        history = window[:self.history_len]
        future = window[self.history_len:]
        
        # Apply random masking if specified
        if self.mask_prob > 0:
            mask = torch.rand_like(window) > self.mask_prob
            window = window * mask.float()
            history = window[:self.history_len]
            future = window[self.history_len:]
        
        return history, future

class WaveletSlidingWindowDataset(SlidingWindowDataset):
    def __init__(self, data_tensor, history_len, predict_len,
                 mask_prob=0.01, wavelet="db4", level=3):
        super().__init__(data_tensor, history_len, predict_len, mask_prob)
        self.wavelet = wavelet
        self.level = level
        # new feature dimension: original_dim * (level + 1)
        self.coeff_dim = data_tensor.shape[1] * (level + 1)

    def __getitem__(self, idx):
        hist, fut = super().__getitem__(idx)  # shape [T, D]
        hist_w = self._decompose_window(hist)
        fut_w = self._decompose_window(fut)
        return hist_w, fut_w

    def _decompose_window(self, win: torch.Tensor) -> torch.Tensor:
        arr = win.numpy()  # [T, D]
        out = []
        for d in range(arr.shape[1]):
            coeffs = modwt_decompose(arr[:, d], self.wavelet, self.level)  # (level+1, T)
            out.append(coeffs.T)  # (T, level+1)
        # Concatenate channels → (T, D*(level+1))
        return torch.from_numpy(np.concatenate(out, axis=1).astype(np.float32))

    def inverse_transform(self, wavelet_data: torch.Tensor) -> torch.Tensor:
        """
        Convert wavelet domain predictions back to time domain.
        Input shape: [T, D*(level+1)]
        Output shape: [T, D]
        """
        arr = wavelet_data.numpy()
        T = arr.shape[0]
        D = self.data.shape[1]
        coeffs_per_dim = self.level + 1
        
        reconstructed = []
        for d in range(D):
            # Extract coefficients for this dimension
            start_idx = d * coeffs_per_dim
            end_idx = (d + 1) * coeffs_per_dim
            dim_coeffs = arr[:, start_idx:end_idx].T  # [(level+1), T]
            
            # Reconstruct this dimension
            rec = imodwt_reconstruct(dim_coeffs, self.wavelet)
            reconstructed.append(rec)
            
        # Stack reconstructed dimensions
        return torch.from_numpy(np.stack(reconstructed, axis=1).astype(np.float32))
