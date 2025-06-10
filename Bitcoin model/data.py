import os
import pandas as pd
import numpy as np
import torch
from functools import reduce

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


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, history_len, predict_len, mask_prob=0.01):
        self.data = data_tensor
        self.history_len = history_len
        self.predict_len = predict_len
        self.mask_prob = mask_prob
        self.total_len = history_len + predict_len
        self.max_idx = len(self.data) - self.total_len
        if self.max_idx <= 0:
            raise ValueError(f"Insufficient data: dataset has {len(self.data)} rows but requires at least {self.total_len}.")

    def __len__(self):
        return min(self.max_idx, 10000)  # default limit

    def __getitem__(self, _):
        i = torch.randint(0, self.max_idx, (1,)).item()
        window = self.data[i:i + self.total_len]

        if self.mask_prob > 0:
            mask = (torch.rand_like(window) > self.mask_prob).float()
            window = window * mask

        return window[:self.history_len], window[self.history_len:]
